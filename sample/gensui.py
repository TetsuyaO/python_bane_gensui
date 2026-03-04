import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#ちょっとテストしてみた。

def FDM(init_x, init_v, init_t, gamma, omega ,dt, T):
    '''
    init_x  :マスの初期位置
    init_v  :マスの初期速度
    init_t  :マスの初期時刻
    gamma   :ダンパーの減衰係数 / (2.0 * マスの質量) -> マス質量は1.0と仮定
    omega   :周波数
    '''

    # parameter
    x = init_x
    v = init_v
    t = init_t

    g, w0 = gamma, omega
    num_iter = int(T/dt)

    alpha = np.arctan(-1*g/np.sqrt(w0**2 - g**2))
    a = np.sqrt(w0**2 * x**2 / (w0**2 - g**2))

    # data array
    t_array = []
    x_array = []
    v_array = []
    x_analytical_array = []
    diff_array = []

    # time step loop
    for i in range(num_iter):
        fx = v
        fv = -1*w0**2 * x - 2*g * v
        x = x + dt * fx
        v = v + dt * fv
        t = t + dt
        x_a = a * np.exp(-1*g * t) * np.cos(np.sqrt(w0**2 - g**2) * t + alpha)
        diff = x_a - x

        t_array.append(t)
        x_array.append(x)
        x_analytical_array.append(x_a)
        v_array.append(v)
        diff_array.append(diff)

    return t_array, x_array, v_array, x_analytical_array, diff_array

def analytical_solution(g, w0, t):
    '''
    g   :ダンパーの減衰係数 / (2.0 * マスの質量) -> マス質量は1.0と仮定
    w0  :周波数
    t   :tf.linespace
    '''
    assert g <= w0
    w = np.sqrt(w0**2-g**2)
    phi = np.arctan(-g/w)
    A = 1/(2*np.cos(phi))
    cos = tf.math.cos(phi+w*t)
    sin = tf.math.sin(phi+w*t)
    exp = tf.math.exp(-g*t)
    x  = exp*2*A*cos
    return x

def MLP(n_input, n_output, n_neuron, n_layer, act_fn='tanh'):
    tf.random.set_seed(1234)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(
            units=n_neuron,
            activation=act_fn,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            input_shape=(n_input,),
            name='H1')
    ])
    for i in range(n_layer-1):
        model.add(
            tf.keras.layers.Dense(
                units=n_neuron,
                activation=act_fn,
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                name='H{}'.format(str(i+2))
            ))
    model.add(
        tf.keras.layers.Dense(
            units=n_output,
            name='output'
        ))
    return model

class EarlyStopping:

    def __init__(self, patience=10, verbose=0):
        '''
        Parameters:
            patience(int): 監視するエポック数(デフォルトは10)
            verbose(int): 早期終了の出力フラグ
                          出力(1),出力しない(0)
        '''

        self.epoch = 0 # 監視中のエポック数のカウンターを初期
        self.pre_loss = float('inf') # 比較対象の損失を無限大'inf'で初期化
        self.patience = patience # 監視対象のエポック数をパラメーターで初期化
        self.verbose = verbose # 早期終了メッセージの出力フラグをパラメーターで初期化

    def __call__(self, current_loss):
        '''
        Parameters:
            current_loss(float): 1エポック終了後の検証データの損失
        Return:
            True:監視回数の上限までに前エポックの損失を超えた場合
            False:監視回数の上限までに前エポックの損失を超えない場合
        '''

        if self.pre_loss < current_loss: # 前エポックの損失より大きくなった場合
            self.epoch += 1 # カウンターを1増やす

            if self.epoch > self.patience: # 監視回数の上限に達した場合
                if self.verbose:  # 早期終了のフラグが1の場合
                    print('early stopping')
                return True # 学習を終了するTrueを返す

        else: # 前エポックの損失以下の場合
            self.epoch = 0 # カウンターを0に戻す
            self.pre_loss = current_loss # 損失の値を更新す

        return False


class DataDrivenNNs():

    def __init__(self, n_input, n_output, n_neuron, n_layer, epochs, act_fn='tanh'):
        '''
        n_input   : インプット数
        n_output   : アウトプット数
        n_neuron   : 隠れ層のユニット数
        n_layer   : 隠れ層の層数
        act_fn   : 活性化関数
        epochs   : エポック数
        '''
        self.n_input = n_input
        self.n_output = n_output
        self.n_neuron = n_neuron
        self.n_layer = n_layer
        self.epochs = epochs
        self.act_fn = act_fn

    def build(self, optimizer, loss_fn, early_stopping):
        self._model = MLP(self.n_input, self.n_output, self.n_neuron, self.n_layer, self.act_fn)
        self._optimizer = optimizer
        self._loss_fn = loss_fn
        self._early_stopping = early_stopping
        return self

    def train_step(self, t_data, x_data):
        with tf.GradientTape() as tape:
            x_pred = self._model(t_data)
            loss = self._loss_fn(x_pred,x_data)
        self._gradients = tape.gradient(loss,self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(self._gradients, self._model.trainable_variables))
        self._loss_values.append(loss)
        return self

    def train(self, t, x, t_data, x_data):

        self._loss_values = []

        for i in range(self.epochs):
            self.train_step(t_data, x_data)
            if self._early_stopping(self._loss_values[-1]):
                break


class PhysicsInformedNNs():

    def __init__(self, n_input, n_output, n_neuron, n_layer, epochs, act_fn='tanh'):
        '''
        n_input   : インプット数
        n_output   : アウトプット数
        n_neuron   : 隠れ層のユニット数
        n_layer   : 隠れ層の層数
        act_fn   : 活性化関数
        epochs   : エポック数
        '''
        self.n_input = n_input
        self.n_output = n_output
        self.n_neuron = n_neuron
        self.n_layer = n_layer
        self.epochs = epochs
        self.act_fn = act_fn

    def build(self, optimizer, loss_fn, early_stopping):
        self._model = MLP(self.n_input, self.n_output, self.n_neuron, self.n_layer, self.act_fn)
        self._optimizer = optimizer
        self._loss_fn = loss_fn
        self._early_stopping = early_stopping
        return self

    def train_step(self, t_data, x_data, t_pinn, c, k):
        with tf.GradientTape() as tape_total:
            tape_total.watch(self._model.trainable_variables)
            x_pred = self._model(t_data)
            loss1 = self._loss_fn(x_pred, x_data)
            loss1 = tf.cast(loss1, dtype=tf.float32)

            with tf.GradientTape() as tape2:
                tape2.watch(t_pinn)
                with tf.GradientTape() as tape1:
                    tape1.watch(t_pinn)
                    x_pred_pinn = self._model(t_pinn)
                dx_dt = tape1.gradient(x_pred_pinn, t_pinn)
            dx_dt2 = tape2.gradient(dx_dt, t_pinn)

            dx_dt  = tf.cast(dx_dt, dtype=tf.float32)
            dx_dt2 = tf.cast(dx_dt2, dtype=tf.float32)
            x_pred_pinn = tf.cast(x_pred_pinn, dtype=tf.float32)

            loss_physics = dx_dt2 + c * dx_dt + k * x_pred_pinn
            loss2 = 5.0e-4 * self._loss_fn(loss_physics, tf.zeros_like(loss_physics))
            loss2 = tf.cast(loss2, dtype=tf.float32)

            loss = loss1 + loss2

        self._optimizer.minimize(loss, self._model.trainable_variables, tape=tape_total)
        self._loss_values.append(loss)
        return self

    def train(self, t, x, t_data, x_data, t_pinn, c, k):
        self._loss_values = []
        for i in range(self.epochs):
            self.train_step(t_data, x_data, t_pinn, c, k)
            if self._early_stopping(self._loss_values[-1]):
                break

def visualize_results(gt_001, gx_001, gx_a, t, x, DDNNs, PINNs):
    # Set figure style
    plt.style.use('default')  # seabornの代わりにdefaultスタイルを使用
    
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
    ax2.scatter(t_data, x_data, c='b', s=100, label='Training Data')
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
    ax3.scatter(t_data, x_data, c='b', s=100, label='Training Data')
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

if __name__ == "__main__":

    ################## Finite difference method ##################

    _,_,_,_,diff_00005 = FDM(1.0, 0.0, 0.0, 2.0, 20.0, 0.00005, 1.0)
    gt_001,gx_001,_,gx_a,diff_001 = FDM(1.0, 0.0, 0.0, 2.0, 20.0, 0.001, 1.0)
    gt_005,gx_005,_,_,diff_005 = FDM(1.0, 0.0, 0.0, 2.0, 20.0, 0.005, 1.0)
    gt_01,gx_01,_,_,diff_015 = FDM(1.0, 0.0, 0.0, 2.0, 20.0, 0.015, 1.0)

    ################ Data-driven neural networks ################

    g, w0 = 2, 20
    c, k = 2*g, w0**2

    t = tf.linspace(0,1,500)
    t = tf.reshape(t,[-1,1])

    x = analytical_solution(g, w0, t)
    x = tf.reshape(x,[-1,1])

    # Data points
    datapoint_list = [i for i in range(0,300,20)]
    t_data = tf.gather(t, datapoint_list)
    x_data = tf.gather(x, datapoint_list)

    DDNNs = DataDrivenNNs(1,1,32,4,5000)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = tf.keras.losses.MeanSquaredError()
    early_stopping = EarlyStopping(patience=200,verbose=1)
    DDNNs.build(optimizer, loss_fn, early_stopping)
    DDNNs.train(t,x,t_data,x_data)

    ############## Physics-informed neural networks ##############

    t_pinn = tf.linspace(0,1,30)
    t_pinn = tf.reshape(t_pinn,[-1,1])

    # Random data points
    random_list = [0,35,50,110,300]
    t_data = tf.gather(t, random_list)
    x_data = tf.gather(x, random_list)

    PINNs = PhysicsInformedNNs(1,1,32,4,50000)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = tf.keras.losses.MeanSquaredError()
    early_stopping = EarlyStopping(patience=200,verbose=1)
    PINNs.build(optimizer, loss_fn, early_stopping)
    PINNs.train(t, x, t_data, x_data, t_pinn, c, k)
    # 結果の可視化
    visualize_results(gt_001, gx_001, gx_a, t, x, DDNNs, PINNs)