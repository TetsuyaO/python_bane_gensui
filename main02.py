import tensorflow as tf
from data_generator import generate_training_data
from training import DataDrivenNNs, PhysicsInformedNNs, EarlyStopping
import matplotlib.pyplot as plt
import os
import numpy as np

# 日本語フォント設定
plt.rcParams['font.family'] = 'IPAexGothic'
plt.rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化け防止
plt.rcParams['font.size'] = 12  # フォントサイズの設定


class ModelManager:
    def __init__(self, model_type, n_neuron=32, n_layer=4):
        self.model_type = model_type
        if model_type == "DDNN":
            self.model = DataDrivenNNs(1, 1, n_neuron, n_layer, 1)
        elif model_type == "PINN":
            self.model = PhysicsInformedNNs(1, 1, n_neuron, n_layer, 1)
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.early_stopping = EarlyStopping(patience=200, verbose=0)
        self.model.build(self.optimizer, self.loss_fn, self.early_stopping)
    
    def save_model(self, path):
        """モデルの保存"""
        if not os.path.exists(path):
            os.makedirs(path)
        self.model._model.save_weights(f"{path}/{self.model_type}_weights.h5")
    
    def load_model(self, path):
        """モデルの読み込み"""
        weights_path = f"{path}/{self.model_type}_weights.h5"
        if os.path.exists(weights_path):
            self.model._model.load_weights(weights_path)
            return True
        return False

def visualize_training_progress(t, x_analytical, t_data_ddnn, x_data_ddnn, 
                              t_data_pinn, x_data_pinn, n_neuron=32, n_layer=4,
                              steps_to_show=[0, 100, 500, 1000, 5000],
                              save_models=True,
                              model_path='./saved_models'):
    """学習過程を可視化する関数（最適化版）
    
    Args:
        t: 時間データ（全範囲）
        x_analytical: 減衰方程式
        t_data_ddnn: DDNNの学習用時間データ
        x_data_ddnn: DDNNの学習用位置データ
        t_data_pinn: PINNの学習用時間データ
        x_data_pinn: PINNの学習用位置データ
        n_neuron: ニューロン数
        n_layer: 層数
        steps_to_show: 表示するステップ数のリスト
        save_models: モデルを保存するかどうか
        model_path: モデルの保存パス
    """
    
    # モデルマネージャーの初期化
    ddnn_manager = ModelManager("DDNN", n_neuron, n_layer)
    pinn_manager = ModelManager("PINN", n_neuron, n_layer)
    
    # 既存のモデルをロード
    if ddnn_manager.load_model(model_path) and pinn_manager.load_model(model_path):
        print("既存のモデルを読み込みました。予測を実行します。")
        # 予測の可視化
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # DDNN予測の可視化
        x_pred_ddnn = ddnn_manager.model._model(t)
        ax1.plot(t, x_analytical, 'r--', label='減衰方程式', linewidth=2)
        ax1.plot(t, x_pred_ddnn, 'g-', label='DDNN予測', linewidth=2)
        ax1.scatter(t_data_ddnn, x_data_ddnn, c='b', s=50, label='学習データ')
        ax1.set_title('DDNN予測結果', fontsize=12)
        ax1.set_ylabel('位置 (x)', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(fontsize=10, loc='upper right')
        
        # PINN予測の可視化
        x_pred_pinn = pinn_manager.model._model(t)
        ax2.plot(t, x_analytical, 'r--', label='減衰方程式', linewidth=2)
        ax2.plot(t, x_pred_pinn, 'g-', label='PINN予測', linewidth=2)
        ax2.scatter(t_data_pinn, x_data_pinn, c='b', s=50, label='学習データ')
        ax2.set_title('PINN予測結果', fontsize=12)
        ax2.set_xlabel('時間 (t)', fontsize=12)
        ax2.set_ylabel('位置 (x)', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(fontsize=10, loc='upper right')
        
        plt.tight_layout()
        plt.show()
        return
    
    print("モデルの学習を開始します。")
    n_steps = len(steps_to_show)
    fig, axes = plt.subplots(2, n_steps, figsize=(20, 8))
    
    # 累積ステップ数を計算
    step_differences = [steps_to_show[i] - steps_to_show[i-1] if i > 0 else steps_to_show[0] 
                       for i in range(len(steps_to_show))]
    
    # プログレスバーの設定
    total_steps = sum(step_differences)
    current_step = 0
    
    for col, (step, step_diff) in enumerate(zip(steps_to_show, step_differences)):
        print(f"\nTraining for {step_diff} steps (Total progress: {current_step}/{total_steps})")
        
        # 差分のステップ数だけ学習
        for i in range(step_diff):
            # DDNNの学習
            ddnn_loss = ddnn_manager.model.train_step(t_data_ddnn, x_data_ddnn)
            
            # PINNの学習
            pinn_loss = pinn_manager.model.train_step(t_data_pinn, x_data_pinn, t, 4.0, 400.0)
            
            # 進捗表示（100ステップごと）
            if (i + 1) % 100 == 0:
                print(f"Step {i+1}/{step_diff}, DDNN Loss: {ddnn_loss:.6f}, PINN Loss: {pinn_loss:.6f}")
            
            # 早期終了の確認
            if ddnn_manager.early_stopping.should_stop() or pinn_manager.early_stopping.should_stop():
                print(f"Early stopping triggered at step {current_step + i + 1}")
                break
        
        current_step += step_diff
        
        # DDNNの予測と可視化
        x_pred_ddnn = ddnn_manager.model._model(t)
        axes[0, col].plot(t, x_analytical, 'r--', label='減衰方程式', linewidth=2)
        axes[0, col].plot(t, x_pred_ddnn, 'g-', label='DDNN予測', linewidth=2)
        axes[0, col].scatter(t_data_ddnn, x_data_ddnn, c='b', s=50, label='学習データ')
        axes[0, col].set_title(f'DDNN (Step {step})', fontsize=12)
        if col == 0:
            axes[0, col].set_ylabel('位置 (x)', fontsize=12)
        axes[0, col].grid(True, linestyle='--', alpha=0.7)
        if col == n_steps-1:
            axes[0, col].legend(fontsize=10, loc='upper right')
        
        # PINNの予測と可視化
        x_pred_pinn = pinn_manager.model._model(t)
        axes[1, col].plot(t, x_analytical, 'r--', label='減衰方程式', linewidth=2)
        axes[1, col].plot(t, x_pred_pinn, 'g-', label='PINN予測', linewidth=2)
        axes[1, col].scatter(t_data_pinn, x_data_pinn, c='b', s=50, label='学習データ')
        axes[1, col].set_title(f'PINN (Step {step})', fontsize=12)
        if col == 0:
            axes[1, col].set_ylabel('位置 (x)', fontsize=12)
        axes[1, col].grid(True, linestyle='--', alpha=0.7)
        if col == n_steps-1:
            axes[1, col].legend(fontsize=10, loc='upper right')
        
        axes[1, col].set_xlabel('時間 (t)', fontsize=12)
    
    # 最終モデルの保存
    if save_models:
        print("\nモデルを保存中...")
        ddnn_manager.save_model(model_path)
        pinn_manager.save_model(model_path)
        print("モデルの保存が完了しました。")
    
    plt.tight_layout()
    plt.show()

def main():
    # トレーニングデータの生成
    data = generate_training_data()
    
    # 学習過程の可視化とモデル保存
    visualize_training_progress(
        t=data['t'],
        x_analytical=data['x'],
        t_data_ddnn=data['t_data_ddnn'],
        x_data_ddnn=data['x_data_ddnn'],
        t_data_pinn=data['t_data_pinn'],
        x_data_pinn=data['x_data_pinn'],
        steps_to_show=[0, 500, 1000, 1500, 2000, 3000, 15000, 20000],
        save_models=True
    )

if __name__ == "__main__":
    main()