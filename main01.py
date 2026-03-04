import tensorflow as tf
from data_generator import generate_training_data
from matplotlib.animation import FuncAnimation, PillowWriter
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

def visualize_training_progress_animated(t, x_analytical, t_data_ddnn, x_data_ddnn, 
                                       t_data_pinn, x_data_pinn, n_neuron=32, n_layer=4,
                                       step_size=500, total_steps=20000,
                                       save_path='training_animation.gif'):
    """学習過程をGIFアニメーションとして可視化する関数"""
    
    # モデルマネージャーの初期化
    ddnn_manager = ModelManager("DDNN", n_neuron, n_layer)
    pinn_manager = ModelManager("PINN", n_neuron, n_layer)
    
    # フィギュアとaxesの初期化
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle('Neural Network Training Progress', fontsize=14)
    
    # プロットの初期化
    line_analytical1, = ax1.plot(t, x_analytical, 'r--', label='減衰方程式', linewidth=2)
    line_ddnn, = ax1.plot([], [], 'g-', label='DDNN予測', linewidth=2)
    scatter_ddnn = ax1.scatter(t_data_ddnn, x_data_ddnn, c='b', s=50, label='学習データ')
    
    line_analytical2, = ax2.plot(t, x_analytical, 'r--', label='減衰方程式', linewidth=2)
    line_pinn, = ax2.plot([], [], 'g-', label='PINN予測', linewidth=2)
    scatter_pinn = ax2.scatter(t_data_pinn, x_data_pinn, c='b', s=50, label='学習データ')
    
    # グラフの設定
    ax1.set_title('DDNN Prediction', fontsize=12)
    ax1.set_ylabel('Position (x)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=10, loc='upper right')
    
    ax2.set_title('PINN Prediction', fontsize=12)
    ax2.set_xlabel('Time (t)', fontsize=12)
    ax2.set_ylabel('Position (x)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=10, loc='upper right')
    
    # y軸の範囲を設定
    y_min = min(np.min(x_analytical), np.min(x_data_ddnn), np.min(x_data_pinn))
    y_max = max(np.max(x_analytical), np.max(x_data_ddnn), np.max(x_data_pinn))
    margin = (y_max - y_min) * 0.1
    ax1.set_ylim(y_min - margin, y_max + margin)
    ax2.set_ylim(y_min - margin, y_max + margin)
    
    # ステップカウンターテキスト
    step_text = fig.text(0.02, 0.95, '', fontsize=10)
    
    def init():
        """アニメーションの初期化関数"""
        line_ddnn.set_data([], [])
        line_pinn.set_data([], [])
        return line_ddnn, line_pinn, step_text
    
    def update(frame):
        """アニメーションの更新関数"""
        current_step = frame * step_size
        
        # DDNNの学習
        for _ in range(step_size):
            ddnn_manager.model.train_step(t_data_ddnn, x_data_ddnn)
        
        # PINNの学習
        for _ in range(step_size):
            pinn_manager.model.train_step(t_data_pinn, x_data_pinn, t, 4.0, 400.0)
        
        # 予測の更新
        x_pred_ddnn = ddnn_manager.model._model(t)
        x_pred_pinn = pinn_manager.model._model(t)
        
        # プロットの更新
        line_ddnn.set_data(t, x_pred_ddnn)
        line_pinn.set_data(t, x_pred_pinn)
        
        # ステップ数の更新
        step_text.set_text(f'Training Step: {current_step}')
        
        return line_ddnn, line_pinn, step_text
    
    # アニメーションの作成
    frames = total_steps // step_size
    anim = FuncAnimation(fig, update, frames=frames, init_func=init,
                        interval=200, blit=True)
    
    # GIFとして保存
    writer = PillowWriter(fps=5)
    anim.save(save_path, writer=writer)
    
    plt.close()
    print(f"Animation saved as {save_path}")

def main():
    # トレーニングデータの生成
    data = generate_training_data()
    
    # アニメーション作成
    visualize_training_progress_animated(
        t=data['t'],
        x_analytical=data['x'],
        t_data_ddnn=data['t_data_ddnn'],
        x_data_ddnn=data['x_data_ddnn'],
        t_data_pinn=data['t_data_pinn'],
        x_data_pinn=data['x_data_pinn'],
        step_size=500,
        total_steps=20000,
        save_path='training_animation.gif'
    )

if __name__ == "__main__":
    main()