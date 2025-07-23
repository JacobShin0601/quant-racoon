#!/usr/bin/env python3
"""
SPY 종목 강화학습 퀀트매매 시스템 (PyTorch 기반)

핵심 구성요소:
1. TradingEnvironment: 강화학습 환경 (state, action, reward)
2. DQNAgent: Deep Q-Network 에이전트
3. ReplayBuffer: 경험 재생 버퍼
4. RLTradingStrategy: 기존 프레임워크와 통합
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
import pickle
import os

# 기존 프레임워크 import
from .strategies import BaseStrategy
from .calculate_index import StrategyParams, TechnicalIndicators

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)

# Experience 구조체
Experience = namedtuple(
    "Experience", ["state", "action", "reward", "next_state", "done"]
)


class TradingEnvironment:
    """강화학습을 위한 트레이딩 환경 (1일~3주 보유 전략 지원)"""

    def __init__(
        self,
        data: pd.DataFrame,
        window_size: int = 20,
        initial_balance: float = 100000.0,
        transaction_cost: float = 0.001,
        min_holding_days: int = 1,
        max_holding_days: int = 21,
    ):
        """
        Args:
            data: SPY 가격 데이터 (OHLCV + 기술지표)
            window_size: 상태로 사용할 과거 데이터 윈도우 크기
            initial_balance: 초기 자본금
            transaction_cost: 거래 비용 (0.1%)
            min_holding_days: 최소 보유 기간 (1일)
            max_holding_days: 최대 보유 기간 (21일 = 3주)
        """
        self.data = data.copy()
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.min_holding_days = min_holding_days
        self.max_holding_days = max_holding_days

        # 상태 변수들
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0  # -1: 숏, 0: 현금, 1: 롱
        self.position_size = 0  # 포지션 크기
        self.entry_price = 0
        self.entry_step = 0  # 포지션 진입 시점
        self.holding_days = 0  # 현재 보유 기간

        # 성과 추적
        self.portfolio_value_history = []
        self.trade_history = []
        self.max_portfolio_value = initial_balance

        # 상태 특성 정의
        self.feature_columns = self._define_features()
        self.n_features = len(self.feature_columns)
        self.n_actions = 3  # 0: 매도/현금, 1: 보유, 2: 매수

        # 데이터 정규화
        self._normalize_data()

    def _define_features(self) -> List[str]:
        """상태로 사용할 특성들 정의"""
        base_features = ["open", "high", "low", "close", "volume"]

        # 기술적 지표들
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

        # 데이터에 존재하는 특성만 선택
        available_features = []
        for feature in base_features + technical_features:
            if feature in self.data.columns:
                available_features.append(feature)

        logger.info(
            f"사용 가능한 특성: {len(available_features)}개 - {available_features}"
        )
        return available_features

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
        """현재 상태 반환 (과거 window_size 기간의 정규화된 데이터)"""
        if self.current_step < self.window_size:
            # 데이터가 부족한 경우 0으로 패딩
            state = np.zeros((self.window_size, self.n_features))
            available_data = self.data_normalized.iloc[: self.current_step][
                self.feature_columns
            ].values
            state[-len(available_data) :] = available_data
        else:
            # 정상적인 경우
            start_idx = self.current_step - self.window_size
            end_idx = self.current_step
            state = self.data_normalized.iloc[start_idx:end_idx][
                self.feature_columns
            ].values

        # 포지션 정보 및 보유 기간 정보 추가
        position_info = np.array(
            [
                self.position,
                self.position_size / self.initial_balance,
                self.holding_days / self.max_holding_days,  # 정규화된 보유 기간
                (
                    (self.current_step - self.entry_step) / len(self.data)
                    if self.position != 0
                    else 0
                ),  # 진입 후 경과 시간
            ]
        )
        position_expanded = np.tile(position_info, (self.window_size, 1))

        # 상태 합치기: [window_size, n_features + 4]
        state_with_position = np.concatenate([state, position_expanded], axis=1)

        return state_with_position.flatten()  # 1D로 변환

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """한 스텝 실행"""
        if self.current_step >= len(self.data) - 1:
            return self._get_state(), 0, True, {}

        current_price = self.data.iloc[self.current_step]["close"]
        next_price = self.data.iloc[self.current_step + 1]["close"]

        # 행동 실행
        reward = self._execute_action(action, current_price, next_price)

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
            "drawdown": (self.max_portfolio_value - portfolio_value)
            / self.max_portfolio_value,
        }

        return self._get_state(), reward, done, info

    def _execute_action(
        self, action: int, current_price: float, next_price: float
    ) -> float:
        """행동 실행 및 보상 계산"""
        reward = 0

        # 거래 비용 계산
        def calculate_transaction_cost(amount):
            return amount * self.transaction_cost

        if action == 2 and self.position <= 0:  # 매수 (롱 포지션)
            if self.position < 0:  # 숏 포지션이 있으면 먼저 청산
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
                    }
                )

            # 새로운 롱 포지션
            available_amount = self.balance * 0.95  # 95%만 사용 (여유분 확보)
            self.position_size = available_amount / current_price
            transaction_cost = calculate_transaction_cost(available_amount)
            self.balance -= available_amount + transaction_cost
            self.position = 1
            self.entry_price = current_price
            self.entry_step = self.current_step
            self.holding_days = 0  # 새로운 포지션 진입 시 보유 기간 초기화

            # 즉시 보상 (다음 가격 기준)
            price_change = (next_price - current_price) / current_price
            reward = price_change - self.transaction_cost

        elif action == 0 and self.position >= 0:  # 매도 (숏 포지션 또는 현금)
            if self.position > 0:  # 롱 포지션이 있으면 먼저 청산
                # 최소 보유 기간 확인
                if self.holding_days < self.min_holding_days:
                    # 최소 보유 기간 미달 시 페널티
                    reward = -0.02  # 조기 매도 페널티
                else:
                    # 정상 매도
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
                        }
                    )

                    # 즉시 보상
                    price_change = (current_price - self.entry_price) / self.entry_price
                    reward = price_change - self.transaction_cost

                # 포지션 초기화
                self.position = 0
                self.position_size = 0
                self.entry_price = 0
                self.entry_step = 0
                self.holding_days = 0

            # 새로운 숏 포지션 (옵션 - 위험할 수 있음)
            # 현재는 현금 보유만 구현
            if self.position == 0:
                self.position = 0
                self.position_size = 0
                self.entry_price = 0

        elif action == 1:  # 보유
            if self.position != 0:
                # 보유 기간 제약 확인
                if self.holding_days >= self.max_holding_days:
                    # 최대 보유 기간 초과 시 강제 청산
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
                            }
                        )

                        # 강제 청산 페널티
                        price_change = (
                            current_price - self.entry_price
                        ) / self.entry_price
                        reward = (
                            price_change - self.transaction_cost - 0.01
                        )  # 추가 페널티

                        self.position = 0
                        self.position_size = 0
                        self.entry_price = 0
                        self.entry_step = 0
                        self.holding_days = 0
                    else:
                        # 숏 포지션 강제 청산
                        pnl = (self.entry_price - current_price) * abs(
                            self.position_size
                        )
                        transaction_cost = calculate_transaction_cost(
                            abs(self.position_size) * current_price
                        )
                        self.balance += pnl - transaction_cost

                        self.trade_history.append(
                            {
                                "type": "short_close_forced",
                                "price": current_price,
                                "size": self.position_size,
                                "pnl": pnl - transaction_cost,
                                "holding_days": self.holding_days,
                            }
                        )

                        price_change = (
                            self.entry_price - current_price
                        ) / self.entry_price
                        reward = price_change - self.transaction_cost - 0.01

                        self.position = 0
                        self.position_size = 0
                        self.entry_price = 0
                        self.entry_step = 0
                        self.holding_days = 0
                else:
                    # 정상 보유 - 보유 기간에 따른 보상 조정
                    if self.position > 0:
                        price_change = (next_price - current_price) / current_price
                        # 보유 기간이 길수록 보상 증가 (장기 보유 장려)
                        holding_bonus = (
                            min(self.holding_days / self.max_holding_days, 1.0) * 0.05
                        )
                        reward = price_change * (0.1 + holding_bonus)
                    elif self.position < 0:
                        price_change = (current_price - next_price) / current_price
                        holding_bonus = (
                            min(self.holding_days / self.max_holding_days, 1.0) * 0.05
                        )
                        reward = price_change * (0.1 + holding_bonus)
            else:
                # 현금 보유 시 기회비용
                market_return = (next_price - current_price) / current_price
                reward = -abs(market_return) * 0.05  # 기회비용

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


class DQNNetwork(nn.Module):
    """Deep Q-Network 신경망"""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(DQNNetwork, self).__init__()

        # 입력 차원 계산 (window_size * (n_features + position_info))
        self.fc1 = nn.Linear(state_size, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, action_size)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class ReplayBuffer:
    """경험 재생 버퍼"""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        """경험 저장"""
        self.buffer.append(Experience(*args))

    def sample(self, batch_size: int) -> List[Experience]:
        """배치 샘플링"""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """Deep Q-Network 에이전트"""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        memory_size: int = 10000,
        batch_size: int = 32,
    ):

        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

        # 신경망 초기화
        self.q_network = DQNNetwork(state_size, action_size).to(device)
        self.target_network = DQNNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # 타겟 네트워크 초기화
        self.update_target_network()

        # 경험 재생 버퍼
        self.memory = ReplayBuffer(memory_size)

        # 학습 통계
        self.loss_history = []

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """행동 선택 (ε-greedy 정책)"""
        if training and random.random() <= self.epsilon:
            return random.choice(range(self.action_size))

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        """경험 저장"""
        self.memory.push(state, action, reward, next_state, done)

    def replay(self):
        """경험 재생을 통한 학습"""
        if len(self.memory) < self.batch_size:
            return

        # 배치 샘플링
        batch = self.memory.sample(self.batch_size)
        states = torch.FloatTensor([e.state for e in batch]).to(device)
        actions = torch.LongTensor([e.action for e in batch]).to(device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(device)
        dones = torch.BoolTensor([e.done for e in batch]).to(device)

        # 현재 Q값 계산
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # 다음 상태의 최대 Q값 계산 (타겟 네트워크 사용)
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # 손실 계산 및 역전파
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(), 1.0
        )  # 그래디언트 클리핑
        self.optimizer.step()

        # 통계 업데이트
        self.loss_history.append(loss.item())

        # ε 감소
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        """타겟 네트워크 업데이트"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save_model(self, filepath: str):
        """모델 저장"""
        torch.save(
            {
                "q_network_state_dict": self.q_network.state_dict(),
                "target_network_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "loss_history": self.loss_history,
            },
            filepath,
        )

    def load_model(self, filepath: str):
        """모델 로드"""
        checkpoint = torch.load(filepath, map_location=device)
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.loss_history = checkpoint["loss_history"]


class RLTradingStrategy(BaseStrategy):
    """강화학습 기반 트레이딩 전략 (기존 프레임워크 통합)"""

    def __init__(self, params: StrategyParams, model_path: Optional[str] = None):
        super().__init__(params)
        self.strategy_type = "single_asset"
        self.model_path = model_path
        self.agent = None
        self.env = None

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """강화학습 모델을 사용한 매매 신호 생성"""
        if self.agent is None or self.model_path is None:
            logger.warning(
                "강화학습 모델이 로드되지 않았습니다. 랜덤 신호를 생성합니다."
            )
            signals = df.copy()
            signals["signal"] = np.random.choice([-1, 0, 1], size=len(df))
            return signals

        # 환경 초기화
        self.env = TradingEnvironment(df)
        state = self.env.reset()

        signals = []
        done = False

        while not done:
            # 행동 선택 (inference 모드)
            action = self.agent.act(state, training=False)

            # 신호 변환 (0: 매도/-1, 1: 보유/0, 2: 매수/1)
            signal_mapping = {0: -1, 1: 0, 2: 1}
            signal = signal_mapping[action]
            signals.append(signal)

            # 다음 스텝
            state, reward, done, info = self.env.step(action)

        # 신호 DataFrame 생성
        result_df = df.copy()
        # 신호 길이 맞추기
        if len(signals) < len(df):
            signals.extend([0] * (len(df) - len(signals)))
        elif len(signals) > len(df):
            signals = signals[: len(df)]

        result_df["signal"] = signals

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
        dummy_env = TradingEnvironment(dummy_data)
        state_size = len(dummy_env._get_state())

        # 에이전트 초기화 및 모델 로드
        self.agent = DQNAgent(state_size=state_size, action_size=3)
        self.agent.load_model(model_path)
        self.model_path = model_path

        logger.info(f"강화학습 모델 로드 완료: {model_path}")


def prepare_spy_data_for_rl(data_path: str) -> pd.DataFrame:
    """SPY 데이터를 강화학습용으로 전처리"""
    # CSV 파일 읽기
    df = pd.read_csv(data_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # 기술적 지표 계산 (없는 경우)
    params = StrategyParams()
    df_with_indicators = TechnicalIndicators.calculate_all_indicators(df, params)

    # NaN 제거
    df_with_indicators = df_with_indicators.dropna().reset_index(drop=True)

    logger.info(f"SPY 데이터 준비 완료: {len(df_with_indicators)}개 데이터 포인트")
    logger.info(f"사용 가능한 컬럼: {list(df_with_indicators.columns)}")

    return df_with_indicators


def train_rl_agent(
    data: pd.DataFrame,
    episodes: int = 1000,
    model_save_path: str = "models/spy_dqn_model.pth",
    min_holding_days: int = 1,
    max_holding_days: int = 21,
) -> Tuple[DQNAgent, List[float]]:
    """강화학습 에이전트 학습 (1일~3주 보유 전략)"""

    # 환경 초기화
    env = TradingEnvironment(
        data, min_holding_days=min_holding_days, max_holding_days=max_holding_days
    )
    state = env.reset()
    state_size = len(state)

    # 에이전트 초기화
    agent = DQNAgent(state_size=state_size, action_size=3)

    # 학습 통계
    episode_rewards = []
    episode_portfolio_values = []

    logger.info(f"강화학습 훈련 시작: {episodes}개 에피소드")
    logger.info(f"상태 크기: {state_size}, 행동 크기: 3")

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step_count = 0

        while not done:
            # 행동 선택
            action = agent.act(state, training=True)

            # 환경에서 한 스텝 실행
            next_state, reward, done, info = env.step(action)

            # 경험 저장
            agent.remember(state, action, reward, next_state, done)

            # 학습 (충분한 경험이 쌓인 후)
            if len(agent.memory) > agent.batch_size:
                agent.replay()

            state = next_state
            total_reward += reward
            step_count += 1

        episode_rewards.append(total_reward)
        final_portfolio_value = (
            env.portfolio_value_history[-1]
            if env.portfolio_value_history
            else env.initial_balance
        )
        episode_portfolio_values.append(final_portfolio_value)

        # 타겟 네트워크 업데이트 (매 10 에피소드마다)
        if episode % 10 == 0:
            agent.update_target_network()

        # 진행 상황 출력
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_portfolio = np.mean(episode_portfolio_values[-100:])
            logger.info(
                f"에피소드 {episode}: 평균 보상 {avg_reward:.4f}, "
                f"평균 포트폴리오 가치 ${avg_portfolio:.2f}, ε={agent.epsilon:.3f}"
            )

    # 모델 저장
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    agent.save_model(model_save_path)
    logger.info(f"모델 저장 완료: {model_save_path}")

    return agent, episode_rewards


def backtest_rl_strategy(data: pd.DataFrame, agent: DQNAgent) -> Dict[str, Any]:
    """강화학습 전략 백테스팅"""

    env = TradingEnvironment(data)
    state = env.reset()

    # 백테스팅 실행
    portfolio_values = []
    actions_taken = []
    rewards = []

    done = False
    while not done:
        action = agent.act(state, training=False)  # 추론 모드
        state, reward, done, info = env.step(action)

        portfolio_values.append(info["portfolio_value"])
        actions_taken.append(action)
        rewards.append(reward)

    # 성과 지표 계산
    portfolio_series = pd.Series(portfolio_values)
    returns = portfolio_series.pct_change().dropna()

    # Buy & Hold 수익률
    initial_price = data.iloc[0]["close"]
    final_price = data.iloc[-1]["close"]
    buy_hold_return = (final_price - initial_price) / initial_price

    # 전략 수익률
    strategy_return = (portfolio_values[-1] - env.initial_balance) / env.initial_balance

    # 샤프 비율 (연간 기준, 일일 데이터 가정)
    if returns.std() > 0:
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
    else:
        sharpe_ratio = 0

    # 최대 낙폭
    peak = portfolio_series.expanding().max()
    drawdown = (portfolio_series - peak) / peak
    max_drawdown = drawdown.min()

    # 거래 통계
    action_counts = pd.Series(actions_taken).value_counts()

    results = {
        "strategy_return": strategy_return,
        "buy_hold_return": buy_hold_return,
        "excess_return": strategy_return - buy_hold_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "total_trades": len(env.trade_history),
        "portfolio_values": portfolio_values,
        "actions": actions_taken,
        "action_distribution": action_counts.to_dict(),
        "final_portfolio_value": portfolio_values[-1],
    }

    return results
