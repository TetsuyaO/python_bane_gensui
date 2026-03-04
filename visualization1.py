import matplotlib.pyplot as plt

def visualize_results(gt_001, gx_001, gx_a, t, x, DDNNs, PINNs,
                     t_data_ddnn, x_data_ddnn, t_data_pinn, x_data_pinn):
    """
    Visualize the results of FDM, DDNN, and PINN solutions
    
    Parameters:
    gt_001: Time points for FDM solution
    gx_001: Position values from FDM solution
    gx_a: Analytical solution values
    t: Time points for neural network solutions
    x: Analytical solution for neural network comparison
    DDNNs: Data-driven neural network model
    PINNs: Physics-informed neural network model
    t_data_ddnn: Time points used for DDNN training
    x_data_ddnn: Position values used for DDNN training
    t_data_pinn: Time points used for PINN training
    x_data_pinn: Position values used for PINN training
    """
    # Set figure style
    plt.style.use('default')
    
    # Figure 1: Solutions comparison
    fig = plt.figure(figsize=(15, 12))
    
    # Plot 1: Finite Difference Method Results
    ax1 = fig.add_subplot(311)
    ax1.plot(gt_001, gx_001, 'b-', label='FDM (dt=0.001)', linewidth=2)
    ax1.plot(gt_001, gx_a, 'r--', label='Analytical', linewidth=2)
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Position', fontsize=12)
    ax1.set_title('Finite Difference Method vs Analytical Solution', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=10)
    
    # Plot 2: Data-Driven Neural Network Results
    ax2 = fig.add_subplot(312)
    x_pred_ddnn = DDNNs._model(t)
    
    ax2.plot(t, x, 'r--', label='Analytical', linewidth=2)
    ax2.plot(t, x_pred_ddnn, 'g-', label='DDNN Prediction', linewidth=2)
    ax2.scatter(t_data_ddnn, x_data_ddnn, c='b', s=100, label='Training Data')
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Position', fontsize=12)
    ax2.set_title('Data-Driven Neural Network Results', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=10)
    
    # Plot 3: Physics-Informed Neural Network Results
    ax3 = fig.add_subplot(313)
    x_pred_pinn = PINNs._model(t)
    
    ax3.plot(t, x, 'r--', label='Analytical', linewidth=2)
    ax3.plot(t, x_pred_pinn, 'g-', label='PINN Prediction', linewidth=2)
    ax3.scatter(t_data_pinn, x_data_pinn, c='b', s=100, label='Training Data')
    ax3.set_xlabel('Time', fontsize=12)
    ax3.set_ylabel('Position', fontsize=12)
    ax3.set_title('Physics-Informed Neural Network Results', fontsize=14)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Figure 2: Loss histories
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # DDNN Loss
    ax1.plot(DDNNs._loss_values, 'b-', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('DDNN Training Loss History', fontsize=14)
    ax1.set_yscale('log')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # PINN Loss
    ax2.plot(PINNs._loss_values, 'r-', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('PINN Training Loss History', fontsize=14)
    ax2.set_yscale('log')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()