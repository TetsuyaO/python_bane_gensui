
flowchart TB
    subgraph Input
        t_data["時間データ (t_data)"]
        x_data["学習データ (x_data)"]
        t_pinn["PINNでの時間データ (t_pinn)"]
    end

    subgraph NeuralNetwork["ニューラルネットワーク"]
        direction LR
        input_layer["入力層"]
        hidden_layers["隠れ層"]
        output_layer["出力層"]
        input_layer --> hidden_layers
        hidden_layers --> output_layer
    end

    subgraph DataLoss["データ損失の計算"]
        pred["予測値 x_pred"]
        loss1["Loss1 = MSE(x_pred, x_data)"]
    end

    subgraph PhysicsLoss["物理法則損失の計算"]
        direction TB
        x_pinn["x_pred_pinn"]
        dx_dt["dx/dt (速度)"]
        dx_dt2["d²x/dt² (加速度)"]
        physics["物理法則残差\n d²x/dt² + c*dx/dt + k*x"]
        loss2["Loss2 = 5.0e-4 * MSE(物理法則残差, 0)"]
    end

    subgraph TotalLoss["総損失"]
        total["Total Loss = Loss1 + Loss2"]
    end

    t_data --> NeuralNetwork
    NeuralNetwork --> pred
    pred --> loss1
    x_data --> loss1

    t_pinn --> NeuralNetwork
    NeuralNetwork --> x_pinn
    x_pinn --> dx_dt
    dx_dt --> dx_dt2
    x_pinn --> physics
    dx_dt --> physics
    dx_dt2 --> physics
    physics --> loss2

    loss1 --> total
    loss2 --> total