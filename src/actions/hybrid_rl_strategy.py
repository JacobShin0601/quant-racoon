#!/usr/bin/env python3
"""
하이브리드 강화학습 퀀트매매 시스템

전통 퀀트 전략 + 강화학습 조합:
1. 전통 전략 신호를 상태에 포함
2. 강화학습이 전통 신호의 가중치를 학습
3. 최종 행동 = 전통 신호 + RL 보정
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from collections import deque, namedtuple
import random
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import os
import gym
from gym import spaces

# 기존 프레임워크 import
from .strategies import BaseStrategy
from .calculate_index import StrategyParams, TechnicalIndicators
from .backtest_strategies import BacktestEngine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)

# Experience 구조체
Experience = namedtuple(
    "Experience", ["state", "action", "reward", "next_state", "done", "log_prob"]
)


class HybridTradingEnvironment(gym.Env):
    """하이브리드 트레이딩 환경 (전통 신호 + RL)"""

    def __init__(
        self,
        data: pd.DataFrame,
        traditional_signals: pd.DataFrame,
        window_size: int = 20,
        initial_balance: float = 100000.0,
        transaction_cost: float = 0.001,
        min_holding_days: int = 1,
        max_holding_days: int = 21,
        action_type: str = "continuous",
    ):
        """
        Args:
            data: 가격 데이터
            traditional_signals: 전통 전략 신호 (swing 전략 결과)
            action_type: "discrete" or "continuous"
        """
        super().__init__()

        self.data = data.copy()
        self.traditional_signals = traditional_signals.copy()
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.min_holding_days = min_holding_days
        self.max_holding_days = max_holding_days
        self.action_type = action_type

        # 상태 변수들
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0  # -1: 숏, 0: 현금, 1: 롱
        self.position_size = 0
        self.entry_price = 0
        self.entry_step = 0
        self.holding_days = 0

        # 성과 추적
        self.portfolio_value_history = []
        self.trade_history = []
        self.max_portfolio_value = initial_balance

        # 상태 특성 정의
        self.feature_columns = self._define_features()
        self.n_features = len(self.feature_columns)

        # 전통 신호 컬럼
        self.signal_columns = self._define_signal_columns()

        # Gym 환경 설정
        if action_type == "discrete":
            # 이산 행동: [매도, 보유, 매수]
            self.action_space = spaces.Discrete(3)
        else:
            # 연속 행동: [전통 신호 가중치 (-1 ~ 1), RL 보정 (-0.5 ~ 0.5)]
            self.action_space = spaces.Box(
                low=np.array([-1.0, -0.5]), high=np.array([1.0, 0.5]), dtype=np.float32
            )

        # 상태 공간: [window_size * (n_features + signal_features + position_info)]
        state_size = self.window_size * (self.n_features + len(self.signal_columns) + 4)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32
        )

        # 데이터 정규화
        self._normalize_data()

    def _define_features(self) -> List[str]:
        """상태로 사용할 특성들 정의"""
        base_features = ["open", "high", "low", "close", "volume"]

        technical_features = [
            "rsi",
            "macd",
            "macd_signal",
            "macd_histogram",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "ema_short",
            "ema_long",
            "atr",
            "stoch_k",
            "stoch_d",
            "williams_r",
            "cci",
        ]

        available_features = []
        for feature in base_features + technical_features:
            if feature in self.data.columns:
                available_features.append(feature)

        logger.info(
            f"사용 가능한 특성: {len(available_features)}개 - {available_features}"
        )
        return available_features

    def _define_signal_columns(self) -> List[str]:
        """전통 신호 컬럼 정의"""
        signal_columns = []

        # 전통 전략 신호 컬럼들
        potential_signals = ["signal", "position", "weight", "confidence", "score"]

        for col in potential_signals:
            if col in self.traditional_signals.columns:
                signal_columns.append(col)

        logger.info(f"전통 신호 컬럼: {len(signal_columns)}개 - {signal_columns}")
        return signal_columns

    def _normalize_data(self):
        """데이터 정규화 (Z-score)"""
        self.data_normalized = self.data.copy()

        for feature in self.feature_columns:
            if feature in self.data.columns:
                mean = self.data[feature].mean()
                std = self.data[feature].std()
                if std > 0:
                    self.data_normalized[feature] = (self.data[feature] - mean) / std
                else:
                    self.data_normalized[feature] = 0

        # 전통 신호 정규화
        self.signals_normalized = self.traditional_signals.copy()
        for signal_col in self.signal_columns:
            if signal_col in self.traditional_signals.columns:
                mean = self.traditional_signals[signal_col].mean()
                std = self.traditional_signals[signal_col].std()
                if std > 0:
                    self.signals_normalized[signal_col] = (
                        self.traditional_signals[signal_col] - mean
                    ) / std
                else:
                    self.signals_normalized[signal_col] = 0

    def reset(self) -> np.ndarray:
        """환경 초기화"""
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0
        self.position_size = 0
        self.entry_price = 0
        self.entry_step = 0
        self.holding_days = 0

        self.portfolio_value_history = []
        self.trade_history = []
        self.max_portfolio_value = self.initial_balance

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """현재 상태 반환 (가격 데이터 + 전통 신호 + 포지션 정보)"""
        if self.current_step < self.window_size:
            # 초기 데이터 부족 시
            state = np.zeros((self.window_size, self.n_features))
            signals_state = np.zeros((self.window_size, len(self.signal_columns)))

            available_data = self.data_normalized.iloc[: self.current_step][
                self.feature_columns
            ].values
            available_signals = self.signals_normalized.iloc[: self.current_step][
                self.signal_columns
            ].values

            state[-len(available_data) :] = available_data
            signals_state[-len(available_signals) :] = available_signals
        else:
            start_idx = self.current_step - self.window_size
            end_idx = self.current_step

            state = self.data_normalized.iloc[start_idx:end_idx][
                self.feature_columns
            ].values
            signals_state = self.signals_normalized.iloc[start_idx:end_idx][
                self.signal_columns
            ].values

        # 포지션 정보 추가
        position_info = np.array(
            [
                self.position,
                self.position_size / self.initial_balance,
                self.holding_days / self.max_holding_days,
                (
                    (self.current_step - self.entry_step) / len(self.data)
                    if self.position != 0
                    else 0
                ),
            ]
        )
        position_expanded = np.tile(position_info, (self.window_size, 1))

        # 모든 상태 정보 결합
        state_with_signals = np.concatenate(
            [state, signals_state, position_expanded], axis=1
        )
        return state_with_signals.flatten()

    def _get_traditional_signal(self) -> float:
        """현재 시점의 전통 신호 가져오기"""
        if self.current_step >= len(self.traditional_signals):
            return 0.0

        # 전통 신호 결합 (가중 평균)
        current_signals = self.traditional_signals.iloc[self.current_step]

        signal_value = 0.0
        total_weight = 0.0

        if "signal" in current_signals:
            signal_value += current_signals["signal"] * 0.4
            total_weight += 0.4

        if "position" in current_signals:
            signal_value += current_signals["position"] * 0.3
            total_weight += 0.3

        if "weight" in current_signals:
            signal_value += current_signals["weight"] * 0.2
            total_weight += 0.2

        if "confidence" in current_signals:
            signal_value += current_signals["confidence"] * 0.1
            total_weight += 0.1

        if total_weight > 0:
            return signal_value / total_weight
        else:
            return 0.0

    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict]:
        """한 스텝 실행"""
        if self.current_step >= len(self.data) - 1:
            return self._get_state(), 0, True, {}

        current_price = self.data.iloc[self.current_step]["close"]
        next_price = self.data.iloc[self.current_step + 1]["close"]

        # 전통 신호 가져오기
        traditional_signal = self._get_traditional_signal()

        # 행동 실행
        if self.action_type == "discrete":
            reward = self._execute_discrete_action(
                action, traditional_signal, current_price, next_price
            )
        else:
            reward = self._execute_continuous_action(
                action, traditional_signal, current_price, next_price
            )

        # 다음 스텝으로 이동
        self.current_step += 1

        # 보유 기간 업데이트
        if self.position != 0:
            self.holding_days += 1

        # 포트폴리오 가치 업데이트
        portfolio_value = self._calculate_portfolio_value()
        self.portfolio_value_history.append(portfolio_value)

        # 최대 포트폴리오 가치 업데이트
        if portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = portfolio_value

        # 종료 조건 확인
        done = (self.current_step >= len(self.data) - 1) or (
            portfolio_value <= self.initial_balance * 0.5
        )

        info = {
            "portfolio_value": portfolio_value,
            "position": self.position,
            "balance": self.balance,
            "holding_days": self.holding_days,
            "traditional_signal": traditional_signal,
            "drawdown": (self.max_portfolio_value - portfolio_value)
            / self.max_portfolio_value,
        }

        return self._get_state(), reward, done, info

    def _execute_discrete_action(
        self,
        action: int,
        traditional_signal: float,
        current_price: float,
        next_price: float,
    ) -> float:
        """이산 행동 실행 (전통 신호와 조합)"""
        reward = 0

        def calculate_transaction_cost(amount):
            return amount * self.transaction_cost

        # 전통 신호를 고려한 행동 결정
        # action: 0(매도), 1(보유), 2(매수)
        # traditional_signal: -1 ~ 1 (매도 ~ 매수)

        # 전통 신호와 RL 행동의 일치도 계산
        signal_alignment = 0.0
        if action == 2 and traditional_signal > 0:  # RL 매수 + 전통 매수
            signal_alignment = traditional_signal
        elif action == 0 and traditional_signal < 0:  # RL 매도 + 전통 매도
            signal_alignment = -traditional_signal
        elif action == 1 and abs(traditional_signal) < 0.2:  # RL 보유 + 전통 중립
            signal_alignment = 1.0 - abs(traditional_signal)
        else:  # 신호 불일치
            signal_alignment = -abs(traditional_signal) * 0.5

        # 기존 DQN 로직
        if action == 2 and self.position <= 0:  # 매수
            if self.position < 0:  # 숏 포지션 청산
                pnl = (self.entry_price - current_price) * abs(self.position_size)
                transaction_cost = calculate_transaction_cost(
                    abs(self.position_size) * current_price
                )
                self.balance += pnl - transaction_cost

                self.trade_history.append(
                    {
                        "type": "short_close",
                        "price": current_price,
                        "size": self.position_size,
                        "pnl": pnl - transaction_cost,
                        "holding_days": self.holding_days,
                        "traditional_signal": traditional_signal,
                    }
                )

            # 새로운 롱 포지션
            available_amount = self.balance * 0.95
            self.position_size = available_amount / current_price
            transaction_cost = calculate_transaction_cost(available_amount)
            self.balance -= available_amount + transaction_cost
            self.position = 1
            self.entry_price = current_price
            self.entry_step = self.current_step
            self.holding_days = 0

            price_change = (next_price - current_price) / current_price
            reward = price_change - self.transaction_cost + signal_alignment * 0.1

        elif action == 0 and self.position >= 0:  # 매도
            if self.position > 0:
                if self.holding_days < self.min_holding_days:
                    reward = -0.02  # 조기 매도 페널티
                else:
                    sell_amount = self.position_size * current_price
                    pnl = (current_price - self.entry_price) * self.position_size
                    transaction_cost = calculate_transaction_cost(sell_amount)
                    self.balance += sell_amount - transaction_cost

                    self.trade_history.append(
                        {
                            "type": "long_close",
                            "price": current_price,
                            "size": self.position_size,
                            "pnl": pnl - transaction_cost,
                            "holding_days": self.holding_days,
                            "traditional_signal": traditional_signal,
                        }
                    )

                    price_change = (current_price - self.entry_price) / self.entry_price
                    reward = (
                        price_change - self.transaction_cost + signal_alignment * 0.1
                    )

                self.position = 0
                self.position_size = 0
                self.entry_price = 0
                self.entry_step = 0
                self.holding_days = 0

        elif action == 1:  # 보유
            if self.position != 0:
                if self.holding_days >= self.max_holding_days:
                    # 강제 청산
                    if self.position > 0:
                        sell_amount = self.position_size * current_price
                        pnl = (current_price - self.entry_price) * self.position_size
                        transaction_cost = calculate_transaction_cost(sell_amount)
                        self.balance += sell_amount - transaction_cost

                        self.trade_history.append(
                            {
                                "type": "long_close_forced",
                                "price": current_price,
                                "size": self.position_size,
                                "pnl": pnl - transaction_cost,
                                "holding_days": self.holding_days,
                                "traditional_signal": traditional_signal,
                            }
                        )

                        price_change = (
                            current_price - self.entry_price
                        ) / self.entry_price
                        reward = price_change - self.transaction_cost - 0.01

                        self.position = 0
                        self.position_size = 0
                        self.entry_price = 0
                        self.entry_step = 0
                        self.holding_days = 0
                else:
                    # 정상 보유
                    if self.position > 0:
                        price_change = (next_price - current_price) / current_price
                        holding_bonus = (
                            min(self.holding_days / self.max_holding_days, 1.0) * 0.05
                        )
                        reward = (
                            price_change * (0.1 + holding_bonus)
                            + signal_alignment * 0.05
                        )
                    elif self.position < 0:
                        price_change = (current_price - next_price) / current_price
                        holding_bonus = (
                            min(self.holding_days / self.max_holding_days, 1.0) * 0.05
                        )
                        reward = (
                            price_change * (0.1 + holding_bonus)
                            + signal_alignment * 0.05
                        )
            else:
                market_return = (next_price - current_price) / current_price
                reward = -abs(market_return) * 0.05 + signal_alignment * 0.05

        return reward

    def _execute_continuous_action(
        self,
        action: np.ndarray,
        traditional_signal: float,
        current_price: float,
        next_price: float,
    ) -> float:
        """연속 행동 실행 (전통 신호와 조합)"""
        # action[0]: 전통 신호 가중치 (-1 ~ 1)
        # action[1]: RL 보정 (-0.5 ~ 0.5)
        traditional_weight = action[0]
        rl_correction = action[1]

        # 최종 신호 = 전통 신호 * 가중치 + RL 보정
        final_signal = traditional_signal * traditional_weight + rl_correction
        final_signal = np.clip(final_signal, -1.0, 1.0)

        reward = 0

        def calculate_transaction_cost(amount):
            return amount * self.transaction_cost

        # 현재 포지션과 목표 포지션의 차이 계산
        current_position_ratio = (
            self.position_size * current_price / self.initial_balance
        )
        position_change = final_signal - current_position_ratio

        # 포지션 조정
        if abs(position_change) > 0.01:  # 1% 이상 차이날 때만 거래
            if position_change > 0:  # 롱 포지션 증가
                if self.position <= 0:  # 기존 포지션 청산
                    if self.position < 0:
                        pnl = (self.entry_price - current_price) * abs(
                            self.position_size
                        )
                        transaction_cost = calculate_transaction_cost(
                            abs(self.position_size) * current_price
                        )
                        self.balance += pnl - transaction_cost

                        self.trade_history.append(
                            {
                                "type": "short_close",
                                "price": current_price,
                                "size": self.position_size,
                                "pnl": pnl - transaction_cost,
                                "holding_days": self.holding_days,
                                "traditional_signal": traditional_signal,
                                "final_signal": final_signal,
                            }
                        )

                # 새로운 롱 포지션
                target_amount = final_signal * self.initial_balance
                available_amount = min(target_amount, self.balance * 0.95)

                if available_amount > 0:
                    self.position_size = available_amount / current_price
                    transaction_cost = calculate_transaction_cost(available_amount)
                    self.balance -= available_amount + transaction_cost
                    self.position = 1
                    self.entry_price = current_price
                    self.entry_step = self.current_step
                    self.holding_days = 0

                    reward -= self.transaction_cost

            elif position_change < 0:  # 포지션 감소
                if self.position > 0:
                    # 부분 청산
                    sell_ratio = min(abs(position_change), 1.0)
                    sell_amount = self.position_size * current_price * sell_ratio
                    pnl = (
                        (current_price - self.entry_price)
                        * self.position_size
                        * sell_ratio
                    )
                    transaction_cost = calculate_transaction_cost(sell_amount)
                    self.balance += sell_amount - transaction_cost

                    self.position_size *= 1 - sell_ratio
                    if self.position_size < 0.01:  # 거의 없으면 완전 청산
                        self.position = 0
                        self.position_size = 0
                        self.entry_price = 0
                        self.entry_step = 0
                        self.holding_days = 0

                    reward += pnl - transaction_cost

        # 보유 기간 가중치를 고려한 보상
        if self.position != 0:
            price_change = (
                (next_price - current_price) / current_price
                if self.position > 0
                else (current_price - next_price) / current_price
            )

            # 전통 신호와의 일치도 보너스
            signal_alignment = 1.0 - abs(final_signal - traditional_signal)
            holding_bonus = min(self.holding_days / self.max_holding_days, 1.0) * 0.05
            reward += price_change * (0.1 + holding_bonus) + signal_alignment * 0.05
        else:
            # 현금 보유 시 기회비용
            market_return = (next_price - current_price) / current_price
            reward -= abs(market_return) * 0.05

        return reward

    def _calculate_portfolio_value(self) -> float:
        """현재 포트폴리오 가치 계산"""
        current_price = self.data.iloc[self.current_step]["close"]

        if self.position > 0:  # 롱 포지션
            position_value = self.position_size * current_price
            return self.balance + position_value
        elif self.position < 0:  # 숏 포지션
            position_value = (self.entry_price - current_price) * abs(
                self.position_size
            )
            return self.balance + position_value
        else:  # 현금
            return self.balance


class HybridActorCriticNetwork(nn.Module):
    """하이브리드 Actor-Critic 네트워크"""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 128,
        action_type: str = "continuous",
    ):
        super(HybridActorCriticNetwork, self).__init__()

        self.action_type = action_type

        # 공통 레이어
        self.fc1 = nn.Linear(state_size, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)

        # Actor (정책) 네트워크
        if action_type == "discrete":
            self.actor = nn.Linear(hidden_size // 2, action_size)
        else:
            self.actor = nn.Linear(hidden_size // 2, action_size * 2)  # mean, log_std

        # Critic (가치) 네트워크
        self.critic = nn.Linear(hidden_size // 2, 1)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))

        actor_output = self.actor(x)
        critic_output = self.critic(x)

        return actor_output, critic_output

    def get_action(self, state: torch.Tensor, training: bool = True):
        """행동 선택"""
        actor_output, critic_value = self.forward(state)

        if self.action_type == "discrete":
            if training:
                action_probs = F.softmax(actor_output, dim=-1)
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                return action, log_prob, critic_value
            else:
                action = torch.argmax(actor_output, dim=-1)
                return action, None, critic_value
        else:
            mean, log_std = actor_output.chunk(2, dim=-1)
            log_std = torch.clamp(log_std, -20, 2)

            if training:
                std = log_std.exp()
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
                return action, log_prob, critic_value
            else:
                return mean, None, critic_value


class HybridPPOTrainer:
    """하이브리드 PPO 훈련기"""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = 0.0003,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        action_type: str = "continuous",
    ):

        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.action_type = action_type

        # 네트워크 초기화
        self.policy = HybridActorCriticNetwork(
            state_size, action_size, action_type=action_type
        ).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # 훈련 통계
        self.loss_history = []

    def compute_gae(
        self, rewards: List[float], values: List[float], dones: List[bool]
    ) -> List[float]:
        """Generalized Advantage Estimation 계산"""
        advantages = []
        gae = 0

        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i + 1]

            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)

        return advantages

    def update_policy(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        num_epochs: int = 10,
    ):
        """정책 업데이트"""
        for _ in range(num_epochs):
            new_actions, new_log_probs, values = self.policy.get_action(
                states, training=True
            )

            # PPO 클리핑
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                * advantages
            )

            # Actor 손실
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic 손실
            critic_loss = F.mse_loss(values.squeeze(), returns)

            # 전체 손실
            loss = actor_loss + 0.5 * critic_loss

            # 역전파
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

            self.loss_history.append(loss.item())

    def save_model(self, filepath: str):
        """모델 저장"""
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss_history": self.loss_history,
            },
            filepath,
        )

    def load_model(self, filepath: str):
        """모델 로드"""
        checkpoint = torch.load(filepath, map_location=device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.loss_history = checkpoint["loss_history"]


def generate_traditional_signals(data: pd.DataFrame) -> pd.DataFrame:
    """전통 퀀트 전략 신호 생성 (스윙 전략 기반)"""
    logger.info("🔄 전통 퀀트 전략 신호 생성 중...")

    signals = data.copy()

    # RSI 기반 신호
    signals["rsi_signal"] = 0
    signals.loc[signals["rsi"] < 30, "rsi_signal"] = 1  # 과매도
    signals.loc[signals["rsi"] > 70, "rsi_signal"] = -1  # 과매수

    # MACD 기반 신호
    signals["macd_signal"] = 0
    signals.loc[signals["macd"] > signals["macd_signal"], "macd_signal"] = 1
    signals.loc[signals["macd"] < signals["macd_signal"], "macd_signal"] = -1

    # 볼린저 밴드 기반 신호
    signals["bb_signal"] = 0
    signals.loc[signals["close"] < signals["bb_lower"], "bb_signal"] = 1  # 하단 돌파
    signals.loc[signals["close"] > signals["bb_upper"], "bb_signal"] = -1  # 상단 돌파

    # 이동평균 기반 신호
    signals["ma_signal"] = 0
    signals.loc[signals["ema_short"] > signals["ema_long"], "ma_signal"] = 1
    signals.loc[signals["ema_short"] < signals["ema_long"], "ma_signal"] = -1

    # 종합 신호 생성
    signals["signal"] = (
        signals["rsi_signal"] * 0.2
        + signals["macd_signal"] * 0.3
        + signals["bb_signal"] * 0.2
        + signals["ma_signal"] * 0.3
    )

    # 신호 스무딩 (이동평균)
    signals["signal"] = signals["signal"].rolling(window=5, min_periods=1).mean()

    # 포지션 신호 (-1: 숏, 0: 중립, 1: 롱)
    signals["position"] = np.where(
        signals["signal"] > 0.1, 1, np.where(signals["signal"] < -0.1, -1, 0)
    )

    # 신뢰도 점수 (0~1)
    signals["confidence"] = abs(signals["signal"])

    # 가중치 (포지션 크기)
    signals["weight"] = signals["signal"] * signals["confidence"]

    # 종합 점수
    signals["score"] = signals["signal"] * signals["confidence"]

    logger.info("✅ 전통 퀀트 전략 신호 생성 완료")
    return signals


def train_hybrid_rl_agent(
    data: pd.DataFrame,
    traditional_signals: pd.DataFrame,
    episodes: int = 1000,
    model_save_path: str = "models/hybrid_rl_model.pth",
    action_type: str = "continuous",
) -> Tuple[HybridPPOTrainer, List[float]]:
    """하이브리드 강화학습 에이전트 훈련"""

    # 환경 초기화
    env = HybridTradingEnvironment(data, traditional_signals, action_type=action_type)
    state = env.reset()
    state_size = len(state)

    # PPO 훈련기 초기화
    trainer = HybridPPOTrainer(
        state_size,
        env.action_space.shape[0] if action_type == "continuous" else 3,
        action_type=action_type,
    )

    episode_rewards = []

    logger.info(f"🤖 하이브리드 PPO 훈련 시작: {episodes}개 에피소드")
    logger.info(f"📊 상태 크기: {state_size}, 행동 타입: {action_type}")

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        # 에피소드 데이터 수집
        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action, log_prob, value = trainer.policy.get_action(
                state_tensor, training=True
            )

            # 행동 실행
            if action_type == "discrete":
                action_np = action.item()
            else:
                action_np = action.squeeze().detach().cpu().numpy()

            next_state, reward, done, info = env.step(action_np)

            # 데이터 저장
            states.append(state)
            actions.append(action_np)
            log_probs.append(log_prob.item())
            rewards.append(reward)
            values.append(value.item())
            dones.append(done)

            state = next_state
            total_reward += reward

        # GAE 계산
        advantages = trainer.compute_gae(rewards, values, dones)
        returns = [r + g for r, g in zip(rewards, advantages)]

        # 정책 업데이트
        if len(states) > 0:
            states_tensor = torch.FloatTensor(states).to(device)
            actions_tensor = torch.FloatTensor(actions).to(device)
            log_probs_tensor = torch.FloatTensor(log_probs).to(device)
            advantages_tensor = torch.FloatTensor(advantages).to(device)
            returns_tensor = torch.FloatTensor(returns).to(device)

            trainer.update_policy(
                states_tensor,
                actions_tensor,
                log_probs_tensor,
                advantages_tensor,
                returns_tensor,
            )

        episode_rewards.append(total_reward)

        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            logger.info(f"에피소드 {episode}: 평균 보상 {avg_reward:.4f}")

    # 모델 저장
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    trainer.save_model(model_save_path)
    logger.info(f"✅ 하이브리드 PPO 모델 저장 완료: {model_save_path}")

    return trainer, episode_rewards


class HybridRLTradingStrategy(BaseStrategy):
    """하이브리드 강화학습 기반 트레이딩 전략"""

    def __init__(
        self,
        params: StrategyParams,
        model_path: Optional[str] = None,
        action_type: str = "continuous",
    ):
        super().__init__(params)
        self.strategy_type = "hybrid_single_asset"
        self.model_path = model_path
        self.action_type = action_type
        self.trainer = None
        self.env = None

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """하이브리드 강화학습 모델을 사용한 매매 신호 생성"""
        if self.trainer is None or self.model_path is None:
            logger.warning(
                "하이브리드 강화학습 모델이 로드되지 않았습니다. 전통 신호만 사용합니다."
            )
            traditional_signals = generate_traditional_signals(df)
            return traditional_signals

        # 전통 신호 생성
        traditional_signals = generate_traditional_signals(df)

        # 환경 초기화
        self.env = HybridTradingEnvironment(
            df, traditional_signals, action_type=self.action_type
        )
        state = self.env.reset()

        signals = []
        done = False

        while not done:
            # 행동 선택 (inference 모드)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action, _, _ = self.trainer.policy.get_action(state_tensor, training=False)

            # 신호 변환
            if self.action_type == "discrete":
                signal_mapping = {0: -1, 1: 0, 2: 1}
                signal = signal_mapping[action.item()]
            else:
                signal = (
                    action.squeeze().cpu().numpy()[0]
                )  # 첫 번째 차원 (전통 신호 가중치)

            signals.append(signal)

            # 다음 스텝
            state, reward, done, info = self.env.step(
                action.item()
                if self.action_type == "discrete"
                else action.squeeze().cpu().numpy()
            )

        # 신호 DataFrame 생성
        result_df = df.copy()
        if len(signals) < len(df):
            signals.extend([0] * (len(df) - len(signals)))
        elif len(signals) > len(df):
            signals = signals[: len(df)]

        result_df["hybrid_signal"] = signals
        result_df["traditional_signal"] = traditional_signals["signal"]

        return result_df

    def load_trained_model(self, model_path: str):
        """학습된 모델 로드"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

        # 더미 환경으로 state_size 계산
        dummy_data = pd.DataFrame(
            {
                "open": [100],
                "high": [102],
                "low": [98],
                "close": [101],
                "volume": [1000000],
            }
        )
        dummy_signals = pd.DataFrame(
            {
                "signal": [0],
                "position": [0],
                "weight": [0],
                "confidence": [0],
                "score": [0],
            }
        )
        dummy_env = HybridTradingEnvironment(
            dummy_data, dummy_signals, action_type=self.action_type
        )
        state_size = len(dummy_env._get_state())

        # 훈련기 초기화 및 모델 로드
        action_size = 3 if self.action_type == "discrete" else 2
        self.trainer = HybridPPOTrainer(
            state_size, action_size, action_type=self.action_type
        )
        self.trainer.load_model(model_path)

        self.model_path = model_path
        logger.info(f"✅ 하이브리드 강화학습 모델 로드 완료: {model_path}")
