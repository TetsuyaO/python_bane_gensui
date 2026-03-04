import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
import numpy as np
import matplotlib
import japanize_matplotlib
matplotlib.rcParams['font.family'] = "IPAexGothic"

def create_training_animation(t, x_analytical, t_data_ddnn, x_data_ddnn, 
                            t_data_pinn, x_data_pinn, DDNNs, PINNs, 
                            num_frames=50, interval=200):
    """
    学習過程のアニメーションを作成する関数
    
    Parameters:
    num_frames: アニメーションのフレーム数
    interval: フレーム間の間隔(ミリ秒)
    """
    # 初期の重みを保存
    ddnn_weights = [w.numpy() for w in DDNNs._model.get_weights()]
    pinn_weights = [w.numpy() for w in PINNs._model.get_weights()]
    
    # フィギュアとアクシスの設定
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    plt.style.use('default')
    
    def animate(frame):
        ax1.clear()
        ax2.clear()
        
        # ステップ数を指数的に増加させる
        step = int(np.exp(np.log(5000) * frame / num_frames))
        
        # DDNNの学習と描画
        DDNNs._model.set_weights([w.copy() for w in ddnn_weights])
        for i in range(step):
            DDNNs.train_step(t_data_ddnn, x_data_ddnn)
        x_pred_ddnn = DDNNs._model(t)
        
        ax1.plot(t, x_analytical, 'r--', label='解析解', linewidth=2)
        ax1.plot(t, x_pred_ddnn, 'g-', label='DDNN予測', linewidth=2)
        ax1.scatter(t_data_ddnn, x_data_ddnn, c='b', s=50, label='学習データ')
        ax1.set_title(f'DDNN (Step {step})', fontsize=12)
        ax1.set_ylabel('Position (x)', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(fontsize=10)
        ax1.set_ylim(-1, 1)  # y軸の範囲を固定
        
        # PINNの学習と描画
        PINNs._model.set_weights([w.copy() for w in pinn_weights])
        for i in range(step):
            PINNs.train_step(t_data_pinn, x_data_pinn, t, 4.0, 400.0)
        x_pred_pinn = PINNs._model(t)
        
        ax2.plot(t, x_analytical, 'r--', label='解析解', linewidth=2)
        ax2.plot(t, x_pred_pinn, 'g-', label='PINN予測', linewidth=2)
        ax2.scatter(t_data_pinn, x_data_pinn, c='b', s=50, label='学習データ')
        ax2.set_title(f'PINN (Step {step})', fontsize=12)
        ax2.set_ylabel('Position (x)', fontsize=12)
        ax2.set_xlabel('Time (t)', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(fontsize=10)
        ax2.set_ylim(-1, 1)  # y軸の範囲を固定
        
        plt.tight_layout()
    
    # アニメーションの作成
    ani = animation.FuncAnimation(fig, animate, frames=num_frames, 
                                interval=interval, blit=False)
    
    # GIFとして保存
    ani.save('training_progress.gif', writer='pillow')
    
    # 元の重みに戻す
    DDNNs._model.set_weights(ddnn_weights)
    PINNs._model.set_weights(pinn_weights)
    
    plt.show()

def visualize_training_progress(t, x_analytical, t_data_ddnn, x_data_ddnn, 
                              t_data_pinn, x_data_pinn, DDNNs, PINNs, 
                              steps_to_show=[0, 100, 500, 1000, 5000]):
    """
    特定のステップでの学習状態を並べて表示する関数
    """
    n_steps = len(steps_to_show)
    fig, axes = plt.subplots(2, n_steps, figsize=(20, 8))
    plt.style.use('default')
    
    # 初期の重みを保存
    ddnn_weights = [w.numpy() for w in DDNNs._model.get_weights()]
    pinn_weights = [w.numpy() for w in PINNs._model.get_weights()]
    
    for col, step in enumerate(steps_to_show):
        # DDNN
        DDNNs._model.set_weights([w.copy() for w in ddnn_weights])
        for i in range(step):
            DDNNs.train_step(t_data_ddnn, x_data_ddnn)
        
        x_pred_ddnn = DDNNs._model(t)
        axes[0, col].plot(t, x_analytical, 'r--', label='減衰方程式', linewidth=2)
        axes[0, col].plot(t, x_pred_ddnn, 'g-', label='DDNN予測', linewidth=2)
        axes[0, col].scatter(t_data_ddnn, x_data_ddnn, c='b', s=50, label='学習データ')
        axes[0, col].set_title(f'DDNN (Step {step})', fontsize=12)
        if col == 0:
            axes[0, col].set_ylabel('Position (x)', fontsize=12)
        axes[0, col].grid(True, linestyle='--', alpha=0.7)
        if col == n_steps-1:
            axes[0, col].legend(fontsize=10)
        axes[0, col].set_ylim(-1, 1)  # y軸の範囲を固定
        
        # PINN
        PINNs._model.set_weights([w.copy() for w in pinn_weights])
        for i in range(step):
            PINNs.train_step(t_data_pinn, x_data_pinn, t, 4.0, 400.0)
        
        x_pred_pinn = PINNs._model(t)
        axes[1, col].plot(t, x_analytical, 'r--', label='減衰方程式', linewidth=2)
        axes[1, col].plot(t, x_pred_pinn, 'g-', label='PINN予測', linewidth=2)
        axes[1, col].scatter(t_data_pinn, x_data_pinn, c='b', s=50, label='学習データ')
        axes[1, col].set_title(f'PINN (Step {step})', fontsize=12)
        if col == 0:
            axes[1, col].set_ylabel('Position (x)', fontsize=12)
        axes[1, col].grid(True, linestyle='--', alpha=0.7)
        if col == n_steps-1:
            axes[1, col].legend(fontsize=10)
        axes[1, col].set_ylim(-1, 1)  # y軸の範囲を固定
        
        axes[1, col].set_xlabel('Time (t)', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    # 元の重みに戻す
    DDNNs._model.set_weights(ddnn_weights)
    PINNs._model.set_weights(pinn_weights)