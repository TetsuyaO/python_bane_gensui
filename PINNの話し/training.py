import tensorflow as tf

def MLP(n_input, n_output, n_neuron, n_layer, act_fn='tanh'):
    '''多層パーセプトロンモデルを構築する関数
    Args:
        n_input: 入力層のユニット数
        n_output: 出力層のユニット数
        n_neuron: 隠れ層のユニット数
        n_layer: 隠れ層の層数
        act_fn: 活性化関数
    Returns:
        model: MLPモデル
    '''
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
        self.epoch = 0
        self.pre_loss = float('inf')
        self.patience = patience
        self.verbose = verbose

    def __call__(self, current_loss):
        if self.pre_loss < current_loss:
            self.epoch += 1
            if self.epoch > self.patience:
                if self.verbose:
                    print('early stopping')
                return True
        else:
            self.epoch = 0
            self.pre_loss = current_loss
        return False

# データ駆動型ニューラルネットワーク
class DataDrivenNNs:
    def __init__(self, n_input, n_output, n_neuron, n_layer, epochs, act_fn='tanh'):
        """
        データ駆動型ニューラルネットワークの初期化
        
        Parameters:
        n_input: 入力層のニューロン数
        n_output: 出力層のニューロン数
        n_neuron: 隠れ層の1層あたりのニューロン数
        n_layer: 隠れ層の層数
        epochs: 学習エポック数
        act_fn: 活性化関数（デフォルトはtanh）
        """
        self.n_input = n_input
        self.n_output = n_output
        self.n_neuron = n_neuron
        self.n_layer = n_layer
        self.epochs = epochs
        self.act_fn = act_fn
        self._loss_values = []  # 損失値の履歴を保存するリスト

    def build(self, optimizer, loss_fn, early_stopping):
        """
        モデルの構築
        
        Parameters:
        optimizer: 最適化アルゴリズム
        loss_fn: 損失関数
        early_stopping: 早期停止の条件
        """
        self._model = MLP(self.n_input, self.n_output, self.n_neuron, self.n_layer, self.act_fn)
        self._optimizer = optimizer
        self._loss_fn = loss_fn
        self._early_stopping = early_stopping
        return self

    def train_step(self, t_data, x_data):
        """
        1ステップの学習を実行
        
        Parameters:
        t_data: 時間データ
        x_data: 教師データ
        """
        # 勾配の計算
        with tf.GradientTape() as tape:
            x_pred = self._model(t_data)  # モデルの予測
            loss = self._loss_fn(x_pred, x_data)  # 損失の計算
        # 勾配を用いたパラメータの更新
        gradients = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))
        self._loss_values.append(loss)
        return self

    def train(self, t_data, x_data):
        self._loss_values = []
        for i in range(self.epochs):
            self.train_step(t_data, x_data)
            if self._early_stopping(self._loss_values[-1]):
                break

class PhysicsInformedNNs:
    def __init__(self, n_input, n_output, n_neuron, n_layer, epochs, act_fn='tanh'):
        """
        物理情報を考慮したニューラルネットワークの初期化
        （DataDrivenNNsと同じパラメータ構成）
        """
        self.n_input = n_input
        self.n_output = n_output
        self.n_neuron = n_neuron
        self.n_layer = n_layer
        self.epochs = epochs
        self.act_fn = act_fn
        self._loss_values = []

    def build(self, optimizer, loss_fn, early_stopping):
        """
        モデルの構築（DataDrivenNNsと同じ構造）
        """
        self._model = MLP(self.n_input, self.n_output, self.n_neuron, self.n_layer, self.act_fn)
        self._optimizer = optimizer
        self._loss_fn = loss_fn
        self._early_stopping = early_stopping
        return self

    def train_step(self, t_data, x_data, t_pinn, c, k):
        """
        物理法則を考慮した1ステップの学習を実行
        
        Parameters:
        t_data: 時間データ
        x_data: 教師データ
        t_pinn: 物理法則を適用する時間点
        c: 減衰係数
        k: バネ定数
        """
        with tf.GradientTape() as tape_total:
            tape_total.watch(self._model.trainable_variables)
            # データに基づく損失の計算
            x_pred = self._model(t_data)
            loss1 = self._loss_fn(x_pred, x_data)
            loss1 = tf.cast(loss1, dtype=tf.float32)

            # 物理法則に基づく損失の計算
            with tf.GradientTape() as tape2:
                tape2.watch(t_pinn)
                with tf.GradientTape() as tape1:
                    tape1.watch(t_pinn)
                    x_pred_pinn = self._model(t_pinn)
                dx_dt = tape1.gradient(x_pred_pinn, t_pinn)    # 1階微分
            dx_dt2 = tape2.gradient(dx_dt, t_pinn)            # 2階微分

            # データ型の統一
            dx_dt = tf.cast(dx_dt, dtype=tf.float32)
            dx_dt2 = tf.cast(dx_dt2, dtype=tf.float32)
            x_pred_pinn = tf.cast(x_pred_pinn, dtype=tf.float32)

            # 物理法則（運動方程式）に基づく損失
            loss_physics = dx_dt2 + c * dx_dt + k * x_pred_pinn
            loss2 = 5.0e-4 * self._loss_fn(loss_physics, tf.zeros_like(loss_physics))
            loss2 = tf.cast(loss2, dtype=tf.float32)

            # 総損失の計算
            loss = loss1 + loss2

        # 勾配を用いたパラメータの更新
        self._optimizer.minimize(loss, self._model.trainable_variables, tape=tape_total)
        self._loss_values.append(loss)
        return self

    def train(self, t_data, x_data, t_pinn, c, k):
        self._loss_values = []
        for i in range(self.epochs):
            self.train_step(t_data, x_data, t_pinn, c, k)
            if self._early_stopping(self._loss_values[-1]):
                break