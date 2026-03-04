import tensorflow as tf
from data_generator import generate_training_data
from training import DataDrivenNNs, PhysicsInformedNNs, EarlyStopping
import matplotlib.pyplot as plt
import os
from PIL import Image
import io
import numpy as np

# 日本語フォント設定
plt.rcParams['font.family'] = 'IPAexGothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

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
        if not os.path.exists(path):
            os.makedirs(path)
        self.model._model.save_weights(f"{path}/{self.model_type}_weights.h5")
    
    def load_model(self, path):
        weights_path = f"{path}/{self.model_type}_weights.h5"
        if os.path.exists(weights_path):
            self.model._model.load_weights(weights_path)
            return True
        return False

def create_frame(t, x_analytical, t_data_ddnn, x_data_ddnn, t_data_pinn, x_data_pinn, 
                x_pred_ddnn, x_pred_pinn, current_step):
    """単一フレームを生成する補助関数"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # DDNN予測の可視化
    ax1.plot(t, x_analytical, 'r--', label='減衰方程式', linewidth=2)
    ax1.plot(t, x_pred_ddnn, 'g-', label='DDNN予測', linewidth=2)
    ax1.scatter(t_data_ddnn, x_data_ddnn, c='b', s=50, label='学習データ')
    ax1.set_title(f'DDNN Training (Step {current_step})', fontsize=12)
    ax1.set_ylabel('Position (x)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.set_ylim(-1.5, 1.5)
    
    # PINN予測の可視化
    ax2.plot(t, x_analytical, 'r--', label='減衰方程式', linewidth=2)
    ax2.plot(t, x_pred_pinn, 'g-', label='PINN予測', linewidth=2)
    ax2.scatter(t_data_pinn, x_data_pinn, c='b', s=50, label='学習データ')
    ax2.set_title(f'PINN Training (Step {current_step})', fontsize=12)
    ax2.set_xlabel('Time (t)', fontsize=12)
    ax2.set_ylabel('Position (x)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=10, loc='upper right')
    ax2.set_ylim(-1.5, 1.5)
    
    plt.tight_layout()
    
    # プロットをPIL Imageに変換
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return Image.open(buf)

def create_training_animation(t, x_analytical, t_data_ddnn, x_data_ddnn, 
                            t_data_pinn, x_data_pinn, n_neuron=32, n_layer=4,
                            n_frames=50, duration=100, save_path='training_animation.gif'):
    """学習過程をアニメーションGIFとして保存する関数"""
    
    # モデルマネージャーの初期化
    ddnn_manager = ModelManager("DDNN", n_neuron, n_layer)
    pinn_manager = ModelManager("PINN", n_neuron, n_layer)
    
    # フレームを格納するリスト
    frames = []
    
    # 学習ステップ数の計算（等間隔）
    total_steps = 20000
    steps_per_frame = total_steps // n_frames
    
    try:
        # 各フレームの生成
        for frame in range(n_frames):
            # 現在のステップ数
            current_step = frame * steps_per_frame
            
            # モデルの学習
            for _ in range(steps_per_frame):
                ddnn_manager.model.train_step(t_data_ddnn, x_data_ddnn)
                pinn_manager.model.train_step(t_data_pinn, x_data_pinn, t, 4.0, 400.0)
            
            # 予測の実行
            x_pred_ddnn = ddnn_manager.model._model(t)
            x_pred_pinn = pinn_manager.model._model(t)
            
            # フレームの生成と追加
            frame_image = create_frame(
                t, x_analytical, t_data_ddnn, x_data_ddnn, t_data_pinn, x_data_pinn,
                x_pred_ddnn, x_pred_pinn, current_step
            )
            frames.append(frame_image)
        
        # GIFアニメーションの保存
        if frames:
            frames[0].save(
                save_path,
                save_all=True,
                append_images=frames[1:],
                optimize=False,
                duration=duration,
                loop=0
            )
            print(f"Animation saved to {save_path}")
        
    finally:
        # メモリの解放
        for frame in frames:
            frame.close()

def main():
    # トレーニングデータの生成
    data = generate_training_data()
    
    # アニメーションの作成
    create_training_animation(
        t=data['t'],
        x_analytical=data['x'],
        t_data_ddnn=data['t_data_ddnn'],
        x_data_ddnn=data['x_data_ddnn'],
        t_data_pinn=data['t_data_pinn'],
        x_data_pinn=data['x_data_pinn'],
        n_frames=50,
        duration=100,
        save_path='training_animation.gif'
    )

if __name__ == "__main__":
    main()