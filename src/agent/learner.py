#!/usr/bin/env python3
"""
SPY 강화학습 퀀트매매 실행 시스템

지원하는 방법론:
1. DQN (Deep Q-Network) - 이산 행동 공간
2. PPO (Proximal Policy Optimization) - 연속 행동 공간

주요 기능:
1. SPY 데이터 로드 및 전처리
2. 강화학습 모델 훈련 (DQN/PPO)
3. 백테스팅 및 성과 평가
4. 결과 시각화 및 리포트 생성
5. 모델 저장/로드
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import argparse
import json
import warnings
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

# DQN 모듈 import
from src.actions.DQN import (
    TradingEnvironment as DQNTradingEnvironment,
    DQNAgent,
    RLTradingStrategy as DQNRLTradingStrategy,
    prepare_spy_data_for_rl,
    train_rl_agent as train_dqn_agent,
    backtest_rl_strategy as backtest_dqn_strategy,
)

# PPO (Advanced RL) 모듈 import
from src.actions.advanced_rl import (
    ModernTradingEnvironment as PPOTradingEnvironment,
    PPOTrainer,
    ModernRLTradingStrategy as PPORLTradingStrategy,
    train_modern_rl_agent as train_ppo_agent,
)

# 하이브리드 RL 모듈 import
from src.actions.hybrid_rl_strategy import (
    HybridTradingEnvironment,
    HybridPPOTrainer,
    HybridRLTradingStrategy,
    generate_traditional_signals,
    train_hybrid_rl_agent,
)

# 기존 프레임워크 import
from src.actions.calculate_index import StrategyParams
from src.actions.backtest_strategies import BacktestEngine
from src.agent.helper import load_config, Logger
import logging

# 경고 무시
warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# GPU/CPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RLQuantLearner:
    """강화학습 퀀트매매 학습 및 실행 시스템"""

    def __init__(self, config_path: str = "config/config_default.json"):
        """
        Args:
            config_path: 설정 파일 경로
        """
        self.config = load_config(config_path)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 디렉토리 설정
        self.data_dir = "data/macro"
        self.models_dir = "models/reinforcement_learning"
        self.results_dir = f"results/rl_trading_{self.timestamp}"
        self.log_dir = "log"

        # 디렉토리 생성
        for directory in [self.models_dir, self.results_dir, self.log_dir]:
            os.makedirs(directory, exist_ok=True)

        # 결과 저장용
        self.training_results = {}
        self.backtest_results = {}
        self.current_method = None

        logger.info(f"🚀 RLQuantLearner 초기화 완료")
        logger.info(f"📁 모델 저장 경로: {self.models_dir}")
        logger.info(f"📊 결과 저장 경로: {self.results_dir}")

    def load_spy_data(self, symbol: str = "SPY") -> Optional[pd.DataFrame]:
        """SPY 데이터 로드"""
        logger.info(f"📂 {symbol} 데이터 로딩 중...")

        # 데이터 파일 찾기
        data_files = []
        for file in os.listdir(self.data_dir):
            if (
                file.startswith(symbol) or file.startswith(symbol.lower())
            ) and file.endswith(".csv"):
                data_files.append(file)

        if not data_files:
            logger.error(f"❌ {symbol} 데이터 파일을 찾을 수 없습니다.")
            return None

        # 가장 최신 파일 선택
        latest_file = sorted(data_files)[-1]
        file_path = os.path.join(self.data_dir, latest_file)

        logger.info(f"📄 사용할 데이터 파일: {latest_file}")

        try:
            # 데이터 전처리
            data = prepare_spy_data_for_rl(file_path)
            logger.info(f"✅ 데이터 로드 완료: {len(data)}개 데이터 포인트")
            logger.info(f"📅 기간: {data['datetime'].min()} ~ {data['datetime'].max()}")

            return data

        except Exception as e:
            logger.error(f"❌ 데이터 로드 중 오류: {e}")
            return None

    def split_data(
        self, data: pd.DataFrame, train_ratio: float = 0.8
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Train/Test 데이터 분할"""
        split_idx = int(len(data) * train_ratio)

        train_data = data.iloc[:split_idx].copy().reset_index(drop=True)
        test_data = data.iloc[split_idx:].copy().reset_index(drop=True)

        logger.info(f"📊 데이터 분할 완료:")
        logger.info(
            f"  🏋️ Train: {len(train_data)}개 ({train_data['datetime'].min()} ~ {train_data['datetime'].max()})"
        )
        logger.info(
            f"  🧪 Test: {len(test_data)}개 ({test_data['datetime'].min()} ~ {test_data['datetime'].max()})"
        )

        return train_data, test_data

    def train_dqn_model(
        self, train_data: pd.DataFrame, episodes: int = 1000, symbol: str = "SPY"
    ) -> Tuple[DQNAgent, List[float]]:
        """DQN 모델 훈련"""
        logger.info("🤖 DQN 모델 훈련 시작")

        model_save_path = os.path.join(
            self.models_dir, f"{symbol}_dqn_model_{self.timestamp}.pth"
        )

        # 훈련 실행 (1일~3주 보유 전략)
        agent, episode_rewards = train_dqn_agent(
            data=train_data,
            episodes=episodes,
            model_save_path=model_save_path,
            min_holding_days=1,
            max_holding_days=21,
        )

        # 훈련 결과 저장
        self.training_results = {
            "method": "DQN",
            "model_path": model_save_path,
            "episodes": episodes,
            "episode_rewards": episode_rewards,
            "final_epsilon": agent.epsilon,
            "loss_history": agent.loss_history,
            "train_data_points": len(train_data),
        }

        logger.info("✅ DQN 모델 훈련 완료")
        return agent, episode_rewards

    def train_ppo_model(
        self, train_data: pd.DataFrame, episodes: int = 1000, symbol: str = "SPY"
    ) -> Tuple[PPOTrainer, List[float]]:
        """PPO 모델 훈련"""
        logger.info("🤖 PPO 모델 훈련 시작")

        model_save_path = os.path.join(
            self.models_dir, f"{symbol}_ppo_model_{self.timestamp}.pth"
        )

        # 훈련 실행 (연속 행동 공간)
        trainer, episode_rewards = train_ppo_agent(
            data=train_data,
            method="PPO",
            episodes=episodes,
            model_save_path=model_save_path,
            action_type="continuous",
        )

        # 훈련 결과 저장
        self.training_results = {
            "method": "PPO",
            "model_path": model_save_path,
            "episodes": episodes,
            "episode_rewards": episode_rewards,
            "loss_history": trainer.loss_history,
            "train_data_points": len(train_data),
        }

        logger.info("✅ PPO 모델 훈련 완료")
        return trainer, episode_rewards

    def train_hybrid_model(
        self, train_data: pd.DataFrame, episodes: int = 1000, symbol: str = "SPY"
    ) -> Tuple[HybridPPOTrainer, List[float]]:
        """하이브리드 모델 훈련 (전통 전략 + RL)"""
        logger.info("🤖 하이브리드 모델 훈련 시작")

        model_save_path = os.path.join(
            self.models_dir, f"{symbol}_hybrid_model_{self.timestamp}.pth"
        )

        # 전통 신호 생성
        traditional_signals = generate_traditional_signals(train_data)

        # 훈련 실행
        trainer, episode_rewards = train_hybrid_rl_agent(
            data=train_data,
            traditional_signals=traditional_signals,
            episodes=episodes,
            model_save_path=model_save_path,
            action_type="continuous",
        )

        # 훈련 결과 저장
        self.training_results = {
            "method": "HYBRID",
            "model_path": model_save_path,
            "episodes": episodes,
            "episode_rewards": episode_rewards,
            "loss_history": trainer.loss_history,
            "train_data_points": len(train_data),
        }

        logger.info("✅ 하이브리드 모델 훈련 완료")
        return trainer, episode_rewards

    def train_model(
        self,
        train_data: pd.DataFrame,
        episodes: int = 1000,
        method: str = "DQN",
        symbol: str = "SPY",
    ) -> Tuple[Any, List[float]]:
        """강화학습 모델 훈련 (통합 인터페이스)"""
        self.current_method = method

        if method.upper() == "DQN":
            return self.train_dqn_model(train_data, episodes, symbol)
        elif method.upper() == "PPO":
            return self.train_ppo_model(train_data, episodes, symbol)
        elif method.upper() == "HYBRID":
            return self.train_hybrid_model(train_data, episodes, symbol)
        else:
            raise ValueError(f"지원하지 않는 방법론: {method}")

    def evaluate_dqn_model(
        self, agent: DQNAgent, test_data: pd.DataFrame, symbol: str = "SPY"
    ) -> Dict[str, Any]:
        """DQN 모델 평가 (백테스팅)"""
        logger.info("📈 DQN 모델 백테스팅 시작")

        # 백테스팅 실행
        backtest_results = backtest_dqn_strategy(test_data, agent)

        # 기존 프레임워크와 비교를 위한 Buy & Hold 계산
        initial_price = test_data.iloc[0]["close"]
        final_price = test_data.iloc[-1]["close"]
        buy_hold_return = (final_price - initial_price) / initial_price

        # 결과 요약
        results_summary = {
            "method": "DQN",
            "symbol": symbol,
            "test_period": f"{test_data['datetime'].min()} ~ {test_data['datetime'].max()}",
            "test_data_points": len(test_data),
            "strategy_return": backtest_results["strategy_return"],
            "buy_hold_return": buy_hold_return,
            "excess_return": backtest_results["excess_return"],
            "sharpe_ratio": backtest_results["sharpe_ratio"],
            "max_drawdown": backtest_results["max_drawdown"],
            "total_trades": backtest_results["total_trades"],
            "action_distribution": backtest_results["action_distribution"],
            "final_portfolio_value": backtest_results["final_portfolio_value"],
        }

        self.backtest_results = {**backtest_results, **results_summary}

        # 결과 출력
        logger.info("📊 DQN 백테스팅 결과:")
        logger.info(f"  💰 전략 수익률: {results_summary['strategy_return']:.2%}")
        logger.info(f"  📈 Buy & Hold 수익률: {results_summary['buy_hold_return']:.2%}")
        logger.info(f"  ⚡ 초과 수익률: {results_summary['excess_return']:.2%}")
        logger.info(f"  📏 샤프 비율: {results_summary['sharpe_ratio']:.3f}")
        logger.info(f"  📉 최대 낙폭: {results_summary['max_drawdown']:.2%}")
        logger.info(f"  🔄 총 거래 수: {results_summary['total_trades']}")

        return self.backtest_results

    def evaluate_ppo_model(
        self, trainer: PPOTrainer, test_data: pd.DataFrame, symbol: str = "SPY"
    ) -> Dict[str, Any]:
        """PPO 모델 평가 (백테스팅)"""
        logger.info("📈 PPO 모델 백테스팅 시작")

        # PPO 환경으로 백테스팅
        env = PPOTradingEnvironment(test_data, action_type="continuous")
        state = env.reset()

        portfolio_values = []
        actions_taken = []
        total_reward = 0
        done = False

        while not done:
            # 행동 선택 (inference 모드)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action, _, _ = trainer.policy.get_action(state_tensor, training=False)

            # 행동 실행
            action_np = action.squeeze().detach().cpu().numpy()
            next_state, reward, done, info = env.step(action_np)

            # 데이터 수집
            portfolio_values.append(info["portfolio_value"])
            actions_taken.append(action_np[0])  # 포지션 크기만 저장
            total_reward += reward

            state = next_state

        # 성과 계산
        initial_value = 100000
        final_value = portfolio_values[-1] if portfolio_values else initial_value
        strategy_return = (final_value - initial_value) / initial_value

        # Buy & Hold 비교
        initial_price = test_data.iloc[0]["close"]
        final_price = test_data.iloc[-1]["close"]
        buy_hold_return = (final_price - initial_price) / initial_price

        # 샤프 비율 계산
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0

        # 최대 낙폭 계산
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown)

        # 행동 분포
        action_distribution = {}
        for action in actions_taken:
            if action < -0.3:
                action_distribution["short"] = action_distribution.get("short", 0) + 1
            elif action > 0.3:
                action_distribution["long"] = action_distribution.get("long", 0) + 1
            else:
                action_distribution["hold"] = action_distribution.get("hold", 0) + 1

        # 결과 요약
        results_summary = {
            "method": "PPO",
            "symbol": symbol,
            "test_period": f"{test_data['datetime'].min()} ~ {test_data['datetime'].max()}",
            "test_data_points": len(test_data),
            "strategy_return": strategy_return,
            "buy_hold_return": buy_hold_return,
            "excess_return": strategy_return - buy_hold_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "total_trades": len([a for a in actions_taken if abs(a) > 0.1]),
            "action_distribution": action_distribution,
            "final_portfolio_value": final_value,
            "portfolio_values": portfolio_values,
        }

        self.backtest_results = results_summary

        # 결과 출력
        logger.info("📊 PPO 백테스팅 결과:")
        logger.info(f"  💰 전략 수익률: {results_summary['strategy_return']:.2%}")
        logger.info(f"  📈 Buy & Hold 수익률: {results_summary['buy_hold_return']:.2%}")
        logger.info(f"  ⚡ 초과 수익률: {results_summary['excess_return']:.2%}")
        logger.info(f"  📏 샤프 비율: {results_summary['sharpe_ratio']:.3f}")
        logger.info(f"  📉 최대 낙폭: {results_summary['max_drawdown']:.2%}")
        logger.info(f"  🔄 총 거래 수: {results_summary['total_trades']}")

        return self.backtest_results

    def evaluate_hybrid_model(
        self, trainer: HybridPPOTrainer, test_data: pd.DataFrame, symbol: str = "SPY"
    ) -> Dict[str, Any]:
        """하이브리드 모델 평가 (백테스팅)"""
        logger.info("📈 하이브리드 모델 백테스팅 시작")

        # 전통 신호 생성
        traditional_signals = generate_traditional_signals(test_data)

        # 하이브리드 환경으로 백테스팅
        env = HybridTradingEnvironment(
            test_data, traditional_signals, action_type="continuous"
        )
        state = env.reset()

        portfolio_values = []
        actions_taken = []
        traditional_signals_used = []
        total_reward = 0
        done = False

        while not done:
            # 행동 선택 (inference 모드)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action, _, _ = trainer.policy.get_action(state_tensor, training=False)

            # 행동 실행
            action_np = action.squeeze().detach().cpu().numpy()
            next_state, reward, done, info = env.step(action_np)

            # 데이터 수집
            portfolio_values.append(info["portfolio_value"])
            actions_taken.append(action_np[0])  # 전통 신호 가중치
            traditional_signals_used.append(info["traditional_signal"])
            total_reward += reward

            state = next_state

        # 성과 계산
        initial_value = 100000
        final_value = portfolio_values[-1] if portfolio_values else initial_value
        strategy_return = (final_value - initial_value) / initial_value

        # Buy & Hold 비교
        initial_price = test_data.iloc[0]["close"]
        final_price = test_data.iloc[-1]["close"]
        buy_hold_return = (final_price - initial_price) / initial_price

        # 샤프 비율 계산
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0

        # 최대 낙폭 계산
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown)

        # 행동 분포
        action_distribution = {}
        for action in actions_taken:
            if action < -0.3:
                action_distribution["reduce_traditional"] = (
                    action_distribution.get("reduce_traditional", 0) + 1
                )
            elif action > 0.3:
                action_distribution["enhance_traditional"] = (
                    action_distribution.get("enhance_traditional", 0) + 1
                )
            else:
                action_distribution["neutral"] = (
                    action_distribution.get("neutral", 0) + 1
                )

        # 결과 요약
        results_summary = {
            "method": "HYBRID",
            "symbol": symbol,
            "test_period": f"{test_data['datetime'].min()} ~ {test_data['datetime'].max()}",
            "test_data_points": len(test_data),
            "strategy_return": strategy_return,
            "buy_hold_return": buy_hold_return,
            "excess_return": strategy_return - buy_hold_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "total_trades": len([a for a in actions_taken if abs(a) > 0.1]),
            "action_distribution": action_distribution,
            "final_portfolio_value": final_value,
            "portfolio_values": portfolio_values,
            "traditional_signals_used": traditional_signals_used,
        }

        self.backtest_results = results_summary

        # 결과 출력
        logger.info("📊 하이브리드 백테스팅 결과:")
        logger.info(f"  💰 전략 수익률: {results_summary['strategy_return']:.2%}")
        logger.info(f"  📈 Buy & Hold 수익률: {results_summary['buy_hold_return']:.2%}")
        logger.info(f"  ⚡ 초과 수익률: {results_summary['excess_return']:.2%}")
        logger.info(f"  📏 샤프 비율: {results_summary['sharpe_ratio']:.3f}")
        logger.info(f"  📉 최대 낙폭: {results_summary['max_drawdown']:.2%}")
        logger.info(f"  🔄 총 거래 수: {results_summary['total_trades']}")

        return self.backtest_results

    def evaluate_model(
        self,
        model: Any,
        test_data: pd.DataFrame,
        method: str = "DQN",
        symbol: str = "SPY",
    ) -> Dict[str, Any]:
        """모델 평가 (통합 인터페이스)"""
        if method.upper() == "DQN":
            return self.evaluate_dqn_model(model, test_data, symbol)
        elif method.upper() == "PPO":
            return self.evaluate_ppo_model(model, test_data, symbol)
        elif method.upper() == "HYBRID":
            return self.evaluate_hybrid_model(model, test_data, symbol)
        else:
            raise ValueError(f"지원하지 않는 방법론: {method}")

    def visualize_results(self, save_plots: bool = True):
        """결과 시각화"""
        logger.info("📊 결과 시각화 생성 중...")

        method = self.training_results.get("method", "Unknown")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            f"SPY {method} 강화학습 퀀트매매 결과 - {self.timestamp}",
            fontsize=16,
            fontweight="bold",
        )

        # 1. 훈련 과정 (에피소드별 보상)
        if "episode_rewards" in self.training_results:
            axes[0, 0].plot(self.training_results["episode_rewards"], alpha=0.7)
            axes[0, 0].plot(
                pd.Series(self.training_results["episode_rewards"]).rolling(100).mean(),
                color="red",
                linewidth=2,
                label="100-Episode Average",
            )
            axes[0, 0].set_title(f"{method} 훈련 과정: 에피소드별 보상")
            axes[0, 0].set_xlabel("Episode")
            axes[0, 0].set_ylabel("Total Reward")
            axes[0, 0].legend()
            axes[0, 0].grid(True)

        # 2. 포트폴리오 가치 변화
        if "portfolio_values" in self.backtest_results:
            portfolio_values = self.backtest_results["portfolio_values"]

            # Buy & Hold 비교선 추가
            initial_value = 100000  # 초기 자본
            buy_hold_values = [
                initial_value
                * (
                    1
                    + self.backtest_results["buy_hold_return"]
                    * i
                    / len(portfolio_values)
                )
                for i in range(len(portfolio_values))
            ]

            axes[0, 1].plot(portfolio_values, label=f"{method} 전략", linewidth=2)
            axes[0, 1].plot(buy_hold_values, label="Buy & Hold", linewidth=2, alpha=0.7)
            axes[0, 1].set_title("포트폴리오 가치 변화")
            axes[0, 1].set_xlabel("Time Steps")
            axes[0, 1].set_ylabel("Portfolio Value ($)")
            axes[0, 1].legend()
            axes[0, 1].grid(True)

        # 3. 행동 분포
        if "action_distribution" in self.backtest_results:
            action_dist = self.backtest_results["action_distribution"]

            if method == "DQN":
                action_labels = {0: "매도/현금", 1: "보유", 2: "매수"}
                labels = [
                    action_labels.get(k, f"Action {k}") for k in action_dist.keys()
                ]
            else:  # PPO
                labels = list(action_dist.keys())

            values = list(action_dist.values())

            axes[1, 0].pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
            axes[1, 0].set_title(f"{method} 행동 분포")

        # 4. 손실 함수 (학습 과정)
        if "loss_history" in self.training_results:
            loss_history = self.training_results["loss_history"]
            if loss_history:
                axes[1, 1].plot(loss_history, alpha=0.7)
                axes[1, 1].plot(
                    pd.Series(loss_history).rolling(100).mean(),
                    color="red",
                    linewidth=2,
                    label="100-Step Average",
                )
                axes[1, 1].set_title(f"{method} 학습 과정: 손실 함수")
                axes[1, 1].set_xlabel("Training Steps")
                axes[1, 1].set_ylabel("Loss")
                axes[1, 1].legend()
                axes[1, 1].grid(True)

        plt.tight_layout()

        if save_plots:
            plot_path = os.path.join(
                self.results_dir,
                f"{method.lower()}_trading_results_{self.timestamp}.png",
            )
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            logger.info(f"📊 차트 저장: {plot_path}")

        plt.show()

    def save_results(self):
        """결과 저장"""
        logger.info("💾 결과 저장 중...")

        method = self.training_results.get("method", "Unknown")

        # 전체 결과 딕셔너리
        all_results = {
            "timestamp": self.timestamp,
            "method": method,
            "training_results": self.training_results,
            "backtest_results": self.backtest_results,
            "config": self.config,
        }

        # JSON 저장
        results_path = os.path.join(
            self.results_dir, f"{method.lower()}_trading_results_{self.timestamp}.json"
        )
        with open(results_path, "w", encoding="utf-8") as f:
            # NumPy 배열을 리스트로 변환하여 JSON 직렬화 가능하게 만들기
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj

            json.dump(convert_numpy(all_results), f, indent=2, ensure_ascii=False)

        # 요약 텍스트 리포트 생성
        report_path = os.path.join(
            self.results_dir, f"{method.lower()}_trading_report_{self.timestamp}.txt"
        )
        self._generate_text_report(report_path)

        logger.info(f"📄 결과 저장 완료:")
        logger.info(f"  📊 JSON: {results_path}")
        logger.info(f"  📝 리포트: {report_path}")

    def _generate_text_report(self, report_path: str):
        """텍스트 리포트 생성"""
        method = self.training_results.get("method", "Unknown")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"SPY {method} 강화학습 퀀트매매 결과 리포트\n")
            f.write("=" * 80 + "\n")
            f.write(f"생성 시간: {self.timestamp}\n")
            f.write(f"사용 방법론: {method}\n\n")

            # 훈련 결과
            if self.training_results:
                f.write("🤖 훈련 결과\n")
                f.write("-" * 40 + "\n")
                f.write(
                    f"총 에피소드: {self.training_results.get('episodes', 'N/A')}\n"
                )
                if method == "DQN":
                    f.write(
                        f"최종 Epsilon: {self.training_results.get('final_epsilon', 'N/A'):.4f}\n"
                    )
                f.write(
                    f"훈련 데이터 포인트: {self.training_results.get('train_data_points', 'N/A')}\n"
                )
                f.write(
                    f"모델 저장 경로: {self.training_results.get('model_path', 'N/A')}\n\n"
                )

            # 백테스팅 결과
            if self.backtest_results:
                f.write("📈 백테스팅 결과\n")
                f.write("-" * 40 + "\n")
                f.write(
                    f"테스트 기간: {self.backtest_results.get('test_period', 'N/A')}\n"
                )
                f.write(
                    f"테스트 데이터 포인트: {self.backtest_results.get('test_data_points', 'N/A')}\n\n"
                )

                f.write("💰 성과 지표\n")
                f.write(
                    f"  전략 수익률: {self.backtest_results.get('strategy_return', 0):.2%}\n"
                )
                f.write(
                    f"  Buy & Hold 수익률: {self.backtest_results.get('buy_hold_return', 0):.2%}\n"
                )
                f.write(
                    f"  초과 수익률: {self.backtest_results.get('excess_return', 0):.2%}\n"
                )
                f.write(
                    f"  샤프 비율: {self.backtest_results.get('sharpe_ratio', 0):.3f}\n"
                )
                f.write(
                    f"  최대 낙폭: {self.backtest_results.get('max_drawdown', 0):.2%}\n"
                )
                f.write(
                    f"  총 거래 수: {self.backtest_results.get('total_trades', 0)}\n"
                )
                f.write(
                    f"  최종 포트폴리오 가치: ${self.backtest_results.get('final_portfolio_value', 0):,.2f}\n\n"
                )

                # 행동 분포
                if "action_distribution" in self.backtest_results:
                    f.write("🎯 행동 분포\n")
                    if method == "DQN":
                        action_labels = {0: "매도/현금", 1: "보유", 2: "매수"}
                        for action, count in self.backtest_results[
                            "action_distribution"
                        ].items():
                            label = action_labels.get(action, f"Action {action}")
                            f.write(f"  {label}: {count}회\n")
                    else:  # PPO
                        for action, count in self.backtest_results[
                            "action_distribution"
                        ].items():
                            f.write(f"  {action}: {count}회\n")

    def load_trained_model(self, model_path: str, method: str = "DQN") -> Optional[Any]:
        """저장된 모델 로드"""
        try:
            if method.upper() == "DQN":
                strategy = DQNRLTradingStrategy(StrategyParams(), model_path)
                strategy.load_trained_model(model_path)
                logger.info(f"✅ DQN 모델 로드 완료: {model_path}")
                return strategy.agent
            elif method.upper() == "PPO":
                # PPO 모델 로드 로직
                trainer = PPOTrainer(0, 0, action_type="continuous")  # 더미 값
                trainer.load_model(model_path)
                logger.info(f"✅ PPO 모델 로드 완료: {model_path}")
                return trainer
            elif method.upper() == "HYBRID":
                # 하이브리드 모델 로드 로직
                trainer = HybridPPOTrainer(0, 0, action_type="continuous")  # 더미 값
                trainer.load_model(model_path)
                logger.info(f"✅ 하이브리드 모델 로드 완료: {model_path}")
                return trainer
            else:
                raise ValueError(f"지원하지 않는 방법론: {method}")
        except Exception as e:
            logger.error(f"❌ 모델 로드 실패: {e}")
            return None

    def run_full_pipeline(
        self,
        episodes: int = 1000,
        train_ratio: float = 0.8,
        method: str = "DQN",
        symbol: str = "SPY",
    ) -> bool:
        """전체 파이프라인 실행"""
        logger.info(f"🎯 {method} 강화학습 퀀트매매 전체 파이프라인 시작")

        try:
            # 1. 데이터 로드
            data = self.load_spy_data(symbol)
            if data is None:
                return False

            # 2. 데이터 분할
            train_data, test_data = self.split_data(data, train_ratio)

            # 3. 모델 훈련
            model, episode_rewards = self.train_model(
                train_data, episodes, method, symbol
            )

            # 4. 모델 평가
            backtest_results = self.evaluate_model(model, test_data, method, symbol)

            # 5. 결과 시각화
            self.visualize_results(save_plots=True)

            # 6. 결과 저장
            self.save_results()

            logger.info(f"🎉 {method} 전체 파이프라인 완료!")
            return True

        except Exception as e:
            logger.error(f"❌ 파이프라인 실행 중 오류: {e}")
            return False


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="SPY 강화학습 퀀트매매 시스템")
    parser.add_argument("--episodes", type=int, default=1000, help="훈련 에피소드 수")
    parser.add_argument("--symbol", type=str, default="SPY", help="거래할 종목")
    parser.add_argument(
        "--train-ratio", type=float, default=0.8, help="훈련 데이터 비율"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_default.json",
        help="설정 파일 경로",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate", "full"],
        default="full",
        help="실행 모드",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["DQN", "PPO", "HYBRID"],
        default="DQN",
        help="강화학습 방법론",
    )
    parser.add_argument(
        "--model-path", type=str, help="로드할 모델 경로 (evaluate 모드용)"
    )

    args = parser.parse_args()

    # 로거 초기화
    learner = RLQuantLearner(args.config)

    if args.mode == "full":
        # 전체 파이프라인 실행
        success = learner.run_full_pipeline(
            episodes=args.episodes,
            train_ratio=args.train_ratio,
            method=args.method,
            symbol=args.symbol,
        )
        if success:
            print(f"✅ {args.method} 강화학습 퀀트매매 시스템 실행 완료!")
        else:
            print("❌ 실행 중 오류가 발생했습니다.")

    elif args.mode == "train":
        # 훈련만 실행
        data = learner.load_spy_data(args.symbol)
        if data is not None:
            train_data, _ = learner.split_data(data, args.train_ratio)
            model, episode_rewards = learner.train_model(
                train_data, args.episodes, args.method, args.symbol
            )
            print(f"✅ {args.method} 모델 훈련 완료!")

    elif args.mode == "evaluate":
        # 평가만 실행
        if not args.model_path:
            print("❌ evaluate 모드에서는 --model-path가 필요합니다.")
            return

        model = learner.load_trained_model(args.model_path, args.method)
        if model:
            data = learner.load_spy_data(args.symbol)
            if data is not None:
                _, test_data = learner.split_data(data, args.train_ratio)
                backtest_results = learner.evaluate_model(
                    model, test_data, args.method, args.symbol
                )
                learner.visualize_results(save_plots=True)
                print(f"✅ {args.method} 모델 평가 완료!")


if __name__ == "__main__":
    main()
