from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from deepforest import CascadeForestRegressor
import os
import random
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from logging.handlers import RotatingFileHandler
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from skopt import BayesSearchCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
import logging
import time

# 忽略警告
warnings.filterwarnings("ignore")


# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

# 配置日志
logger = logging.getLogger()
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler1 = RotatingFileHandler("feature_selection.log", maxBytes=10 ** 6, backupCount=5, encoding='utf-8')
    handler2 = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger.addHandler(handler1)
    logger.addHandler(handler2)

# 定义基础特征
BASE_FEATURES = ['lat', 'lon', 'CT', 'dm', 'gp', 'pp', 'sh', 'sp', 'te', 'NO2', 'CO', 'nt', 'nv']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"使用设备: {DEVICE}")


# 特征工程类
class FeatureEngineer:
    def __init__(self):
        self.fitted = False

    def fit(self, df):

        self.fitted = True

    def transform(self, df):
        """
        对数据进行特征工程转换。
        """
        if not self.fitted:
            raise RuntimeError("FeatureEngineer 未拟合。请先调用 fit 方法。")

        df = df.copy()

        # 检查原始数据是否包含所有必要的列
        required_columns = [
            'te', 'sh', 'dm', 'nv', 'pp', 'lat', 'lon',
            'CT', 'CO', 'NO2', 'gp', 'sp', 'XCO2', 'time', 'nt',
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"原始数据缺少以下必要列: {missing_columns}")
        else:
            logging.info("原始数据包含所有必要列。")

        # 时间特征
        df['date'] = pd.to_datetime(df['time'].astype(str), format='%Y%m%d', errors='coerce')
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['week'] = df['date'].dt.isocalendar().week
        df['day_of_year'] = df['date'].dt.dayofyear
        df['quarter'] = df['date'].dt.quarter
        df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday

        # 周期性特征
        df['sin_day'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['cos_day'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        df['sin_quarter'] = np.sin(2 * np.pi * df['quarter'] / 4)
        df['cos_quarter'] = np.cos(2 * np.pi * df['quarter'] / 4)
        df['sin_week'] = np.sin(2 * np.pi * df['week'] / 52)
        df['cos_week'] = np.cos(2 * np.pi * df['week'] / 52)
        df['sin_day_of_week'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['cos_day_of_week'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # 交互特征
        df['NO2_temp_interaction'] = df['NO2'] * df['te']
        df['NDVI_GPP_ratio'] = df['nv'] / (df['gp'] + 1e-6)  # 防止除零
        df['elevation_humidity_interaction'] = df['dm'] * df['sh']
        df['dm_sp'] = df['dm'] * df['sp']
        df['NO2_te'] = df['te'] * df['NO2']
        df['CO_te'] = df['CO'] * df['te']

        # 移除因滞后和移动特征产生的缺失值
        # 因为滞后特征被注释掉，下面的代码也可以注释或保留
        df = df.dropna()

        # 检查所有需要的特征是否已创建
        all_required_features = [
            'dm_sp', 'NO2_te', 'CO_te', 'day_of_year', 'quarter', 'year',
            'elevation_humidity_interaction', 'NDVI_GPP_ratio', 'NO2_temp_interaction',
            'sin_day', 'cos_day', 'sin_quarter', 'cos_quarter',
            'sin_week', 'cos_week', 'sin_day_of_week', 'cos_day_of_week',
            'month', 'day', 'week', 'day_of_week', 'lat', 'lon',
            'CT', 'dm', 'gp', 'pp', 'sh', 'sp', 'te', 'wu',
            'wv', 'NO2', 'CO', 'nt'
        ]

        missing_features = [feat for feat in all_required_features if feat not in df.columns]
        if missing_features:
            raise ValueError(f"特征工程后缺少以下特征: {missing_features}")
        else:
            logging.info("所有必要特征已成功创建。")

        return df


# 修改后的监督CNN自编码器（关键修改1）
class SupervisedCNNAutoencoder(nn.Module):
    def __init__(self, input_channels=1, encoding_dim=32, cnn_features=['lat', 'lon', 'CT']):
        super(SupervisedCNNAutoencoder, self).__init__()

        # 计算 feature_length
        feature_length = len(cnn_features)

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),  # 修正为32，匹配前层输出通道
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),

            # nn.Conv1d(32, 64, kernel_size=3, padding=1),
            # nn.BatchNorm1d(64),  # 修正为64，匹配前层输出通道
            # nn.ReLU(),
            # nn.MaxPool1d(2),
            # nn.Dropout(0.3),

            nn.Flatten(),
            # 动态计算输入维度，考虑最大池化后的维度
            nn.Linear(32 * (feature_length // 2), 256),  # 根据卷积后的特征维度来调整
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),

            nn.Linear(256, encoding_dim)
        )

        # 回归层
        self.regression = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        encoded = self.encoder(x)
        xco2_pred = self.regression(encoded)
        return encoded, xco2_pred


# 修改后的训练函数（关键修改3）
def train_supervised_cnn_autoencoder(
        df_train,
        cnn_features=['lat', 'lon', 'CT'],
        encoding_dim=8,
        epochs=100,
        batch_size=128,
        patience=10,
        model_save_path='models/best_supervised_cnn_autoencoder.pth'
):
    # 数据准备
    X_train_cnn = df_train[cnn_features].values
    y_train_cnn = df_train['XCO2'].values

    # 数据标准化
    scaler_cnn = StandardScaler()
    X_train_cnn_scaled = scaler_cnn.fit_transform(X_train_cnn)

    # 修改输入维度为1D卷积（关键修改4）
    feature_length = len(cnn_features)
    X_train_cnn_scaled = X_train_cnn_scaled.reshape(-1, 1, feature_length)

    # 转换为张量
    X_train_tensor = torch.tensor(X_train_cnn_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_cnn, dtype=torch.float32).unsqueeze(1)

    # 划分训练验证集
    X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
        X_train_tensor.numpy(), y_train_tensor.numpy(), test_size=0.2, random_state=42
    )

    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train_sub, dtype=torch.float32),
        torch.tensor(y_train_sub, dtype=torch.float32)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_val_sub, dtype=torch.float32),
        torch.tensor(y_val_sub, dtype=torch.float32)
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    model = SupervisedCNNAutoencoder(
        input_channels=1,
        encoding_dim=encoding_dim,

    ).to(DEVICE)


    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # 训练循环
    best_val_loss = np.inf
    epochs_no_improve = 0
    history = {'train': [], 'val': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            _, outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # 梯度裁剪（关键修改6）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                _, outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        # 计算平均损失
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        history['train'].append(train_loss)
        history['val'].append(val_loss)

        # 学习率调度
        scheduler.step(val_loss)

        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
            logging.info(f'Epoch {epoch + 1}: 最佳验证损失改进为 {best_val_loss:.4f},训练损失为{train_loss:.4f}')
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                logging.info(f'早停触发于第 {epoch + 1} 轮')
                break

        # 打印训练进度
        if (epoch + 1) % 5 == 0:
            logging.info(
                f'Epoch {epoch + 1}/{epochs} | '
                f'训练损失: {train_loss:.4f} | '
                f'验证损失: {val_loss:.4f} | '
                f'学习率: {optimizer.param_groups[0]["lr"]:.6f}'
            )

    # 加载最佳模型
    model.load_state_dict(torch.load(model_save_path))

    # 提取特征
    model.eval()
    with torch.no_grad():
        X_train_tensor = X_train_tensor.to(DEVICE)
        encoded_train, _ = model(X_train_tensor)
        encoded_train = encoded_train.cpu().numpy()

    return encoded_train, scaler_cnn


# 添加LASSO回归特征选择
def lasso_feature_selection_with_bayesian_optimization(X_train, y_train):
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # 设置参数搜索空间
    param_space = {
        'alpha': (1e-6, 1e+1, 'log-uniform'),  # 对数均匀分布范围 [1e-6, 1e+1]
        'max_iter': (1000, 200000)  # 迭代次数范围
    }

    # 使用贝叶斯优化进行超参数搜索
    lasso = Lasso(random_state=42)  # 使用Lasso回归（非LassoCV）
    opt = BayesSearchCV(lasso, param_space, n_iter=50, cv=5, n_jobs=-1, scoring='neg_mean_squared_error',
                        random_state=42)

    # 执行贝叶斯优化
    opt.fit(X_scaled, y_train)

    # 获取最佳参数组合
    best_params = opt.best_params_
    logging.info(f"最佳的LASSO参数: {best_params}")

    # 使用最佳参数重新训练LASSO模型
    best_lasso = Lasso(alpha=best_params['alpha'], max_iter=best_params['max_iter'], random_state=42)
    best_lasso.fit(X_scaled, y_train)

    # 获取非零系数的特征
    selected_features = X_train.columns[(best_lasso.coef_ != 0)].tolist()

    logging.info(f"LASSO选择的特征: {selected_features}")

    return selected_features, best_params


# 贝叶斯优化调参
def optimize_cascade_forest(X_train, y_train):
    """
    使用贝叶斯优化对 CascadeForestRegressor 进行调参。

    参数:
    X_train (DataFrame): 训练集特征数据。
    y_train (Series): 训练集目标变量。

    返回:
    best_params (dict): 调优后的最佳参数。
    """

    print("开始贝叶斯优化......")
    # 设置要优化的参数空间
    param_space = {
        'n_bins': (2, 255),
        'bin_subsample': (200000, 400000),
        'max_layers': (10, 40),
        'n_estimators': (2, 10),
        'n_trees': (100, 600),
    }

    # 定义 CascadeForestRegressor 模型
    cascade_forest = CascadeForestRegressor(n_jobs=-1, partial_mode='true', random_state=42)

    # 使用贝叶斯优化进行超参数搜索  scoring='neg_mean_squared_error',
    bayes_search = BayesSearchCV(cascade_forest, param_space, n_iter=50, cv=5, n_jobs=-1,
                                 scoring=None, random_state=42)
    print("开始执行贝叶斯优化......")
    # 执行贝叶斯优化
    bayes_search.fit(X_train, y_train)

    # 获取最佳参数组合
    best_params = bayes_search.best_params_
    logging.info(f"调优后的最佳参数: {best_params}")

    return best_params


# 训练模型函数
def train_model(df, feature_engineer):
    # 选择特征和目标
    target_column = 'XCO2'

    # 定义所有特征（排除非特征列，例如 'date'）
    excluded_columns = ['date']  # 保留 'XCO2' 和 'time'
    all_feature_columns = [col for col in df.columns if col not in excluded_columns]

    # 划分训练集和测试集
    X = df[all_feature_columns]  # 包含 'XCO2' 和 'time'
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logging.info("\n数据已划分为训练集和测试集。")
    logging.info(f"训练集样本数: {X_train.shape[0]}, 测试集样本数: {X_test.shape[0]}")

    logging.info("\n训练基础模型...")

    # 调用贝叶斯优化调参（直接使用已经定义的 optimize_cascade_forest 函数）
    base_best_params = optimize_cascade_forest(X_train.values, y_train)

    # 使用调优后的参数训练 CascadeForestRegressor 模型
    try:
        base_model = CascadeForestRegressor(
            n_estimators=base_best_params['n_estimators'],
            n_trees=base_best_params['n_trees'],
            max_layers=base_best_params['max_layers'],
            n_bins=base_best_params['n_bins'],
            bin_subsample=base_best_params['bin_subsample'],
            random_state=42,
            n_jobs=-1,
            partial_mode='true'
        )
        base_model.fit(X_train.values, y_train)
        logging.info("模型训练完成。")
    except Exception as e:
        logging.error(f"模型训练失败: {e}")
        raise

    # 模型预测与评估
    logging.info("\n进行模型预测...")
    try:
        base_y_pred = base_model.predict(X_test.values).flatten()  # 预测并转换为一维数组
    except Exception as e:
        logging.error(f"模型预测失败: {e}")
        raise

    try:
        # 对齐 y_test 与 X_test_transformed 的索引
        r2 = r2_score(y_test, base_y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, base_y_pred))
        mae = mean_absolute_error(y_test, base_y_pred)

        logging.info("\nbase评估指标:")
        logging.info(f"R²: {r2:.4f}")
        logging.info(f"RMSE: {rmse:.4f}")
        logging.info(f"MAE: {mae:.4f}")
    except Exception as e:
        logging.error(f"模型评估失败: {e}")
        raise

    # 确保 all_feature_columns are present
    required_columns = [
        'lat', 'lon', 'CT', 'dm', 'gp', 'pp', 'sh', 'sp', 'te',
        'NO2', 'CO', 'nt', 'nv', 'XCO2', 'time'
    ]
    missing_columns = [col for col in required_columns if col not in X_train.columns]
    if missing_columns:
        logging.error(f"训练集缺少以下必要列: {missing_columns}")
        raise ValueError(f"训练集缺少以下必要列: {missing_columns}")
    else:
        logging.info("训练集包含所有必要列。")

    # 在训练集上进行特征工程
    logging.info("\n在训练集上进行特征工程...")
    try:
        feature_engineer.fit(X_train)
        X_train_transformed = feature_engineer.transform(X_train)
        logging.info("训练集特征工程转换完成。")
    except Exception as e:
        logging.error(f"训练集特征工程转换失败: {e}")
        raise

    # 将相同的特征工程应用于测试集
    logging.info("在测试集上进行特征工程...")
    try:
        X_test_transformed = feature_engineer.transform(X_test)
        logging.info("测试集特征工程转换完成。")
    except Exception as e:
        logging.error(f"测试集特征工程转换失败: {e}")
        raise

    # 检查 if 'XCO2' is present
    if 'XCO2' not in X_train_transformed.columns or 'XCO2' not in X_test_transformed.columns:
        logging.error("'XCO2' 列在特征工程后缺失。")
        raise ValueError("'XCO2' 列在特征工程后缺失。")

    # 对齐 y_train 与 X_train_transformed 的索引
    y_train = y_train.loc[X_train_transformed.index]

    df_train_with_y = X_train_transformed.copy()
    if 'XCO2' in df_train_with_y.columns:
        df_train_with_y.drop(columns=['XCO2'], inplace=True)
    df_train_with_y['XCO2'] = y_train
    logging.info("已将 y_train 添加到训练数据中，用于 CNN 特征提取。")

    # 训练并提取监督CNN自编码器特征
    logging.info("\n训练并提取监督CNN自编码器特征...")
    try:
        # 确保 'IDW_XCO2' 已添加
        cnn_features = ['lat', 'lon', 'CT']
        # Check if all cnn_features are present
        missing_cnn_features = [feat for feat in cnn_features if feat not in X_train_transformed.columns]
        if missing_cnn_features:
            raise ValueError(f"缺少CNN特征: {missing_cnn_features}")

        # 修改CNN训练参数
        encoded_train_cnn, scaler_cnn = train_supervised_cnn_autoencoder(
            df_train=df_train_with_y,
            cnn_features=['lat', 'lon', 'CT'],
            encoding_dim=8,
            epochs=100,
            batch_size=128,
            patience=10,
            model_save_path='models/best_supervised_cnn_autoencoder.pth'
        )

        # Add CNN features to the training set
        for i in range(8):
            feature_name = f'AEF_{i + 1}'
            X_train_transformed[feature_name] = encoded_train_cnn[:, i]
        logging.info("CNN特征已添加到训练集。")
    except Exception as e:
        logging.error(f"CNN特征提取失败: {e}")
        raise

    logging.info("提取测试集的CNN编码特征...")
    try:

        X_test_cnn = X_test_transformed[cnn_features].values
        X_test_cnn_scaled = scaler_cnn.transform(X_test_cnn)

        X_test_cnn_scaled = X_test_cnn_scaled.reshape(-1, 1, len(cnn_features))
        X_test_cnn_tensor = torch.tensor(X_test_cnn_scaled, dtype=torch.float32)


        cnn_autoencoder_loaded = SupervisedCNNAutoencoder(
            input_channels=1, encoding_dim=8
        ).to(DEVICE)
        try:
            cnn_autoencoder_loaded.load_state_dict(
                torch.load('models/best_supervised_cnn_autoencoder.pth', map_location=DEVICE)
            )
            cnn_autoencoder_loaded.eval()
            logging.info("已加载最优的 CNN 自编码器模型。")
        except Exception as e:
            logging.error(f"加载最优 CNN 模型失败: {e}")
            raise

        with torch.no_grad():
            # 直接调用模型，获取编码特征
            encoded_test_cnn, _ = cnn_autoencoder_loaded(X_test_cnn_tensor.to(DEVICE))
            encoded_test_cnn = encoded_test_cnn.cpu().numpy()

        # 将提取的 CNN 特征添加到测试集
        for i in range(8):
            feature_name = f'AEF_{i + 1}'
            X_test_transformed[feature_name] = encoded_test_cnn[:, i]
        logging.info("CNN特征已添加到测试集。")
    except Exception as e:
        logging.error(f"测试集CNN特征提取失败: {e}")
        raise

    # 移除特征工程中生成的 'XCO2', 'time', 'date' 列，以避免数据泄漏
    logging.info("\n移除特征工程中生成的 'XCO2', 'time', 'date' 列，以避免数据泄漏...")
    columns_to_remove = ['XCO2', 'time', 'date']
    for col in columns_to_remove:
        if col in X_train_transformed.columns:
            X_train_transformed = X_train_transformed.drop([col], axis=1)
            logging.info(f"已移除训练集中的列: {col}")
        if col in X_test_transformed.columns:
            X_test_transformed = X_test_transformed.drop([col], axis=1)
            logging.info(f"已移除测试集中的列: {col}")

    # 对y进行标准化
    logging.info("\n对目标变量 y 进行标准化...")
    y_scaler = StandardScaler()
    try:
        y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        logging.info("目标变量 y 已标准化。")
    except Exception as e:
        logging.error(f"目标变量 y 标准化失败: {e}")
        raise

    # 数据归一化
    logging.info("对特征进行标准化...")
    scaler = StandardScaler()
    try:
        X_train_scaled = scaler.fit_transform(X_train_transformed)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_transformed.columns,
                                      index=X_train_transformed.index)
        X_test_scaled = scaler.transform(X_test_transformed)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test_transformed.columns, index=X_test_transformed.index)
        logging.info("特征已标准化。")
    except Exception as e:
        logging.error(f"特征标准化失败: {e}")
        raise

    # 前向特征选择
    logging.info("\n开始前向特征选择...")
    try:
        selected_features, not_selected_features = forward_feature_selection_single_split_with_optimization(
            X_train_scaled, y_train_scaled, base_features=BASE_FEATURES, max_features=100
            # 设置 max_features 根据需要
        )
        logging.info(f"\n选择的特征数量: {len(selected_features)}")
    except Exception as e:
        logging.error(f"前向特征选择失败: {e}")
        raise

    # 提取选中的特征
    try:
        X_train_selected = X_train_scaled[selected_features]
        X_test_selected = X_test_scaled[selected_features]
        logging.info("已选择的特征已提取。")
    except Exception as e:
        logging.error(f"提取选择的特征失败: {e}")
        raise

    # 使用LASSO进行特征选择
    logging.info("\n使用LASSO进行特征选择...")
    try:
        # 获取LASSO选择的特征列表
        selected_features_lasso, _ = lasso_feature_selection_with_bayesian_optimization(X_train_selected, y_train)
        logging.info(f"LASSO选择的特征: {selected_features_lasso}")
    except Exception as e:
        logging.error(f"LASSO特征选择失败: {e}")
        raise

    # 检查 selected_features_lasso 是否为空
    if not selected_features_lasso:
        logging.error("LASSO选择的特征列表为空。无法训练模型。")
        raise ValueError("LASSO选择的特征列表为空。无法训练模型。")

    # 使用LASSO选择的特征从训练集和测试集中选择对应的列
    X_train_lasso_selected = X_train_selected[selected_features_lasso]
    X_test_lasso_selected = X_test_selected[selected_features_lasso]

    logging.info("\n训练 CascadeForestRegressor 模型...")

    # 调用贝叶斯优化调参（直接使用已经定义的 optimize_cascade_forest 函数）
    best_params = optimize_cascade_forest(X_train_lasso_selected.values, y_train_scaled)

    # 使用调优后的参数训练 CascadeForestRegressor 模型
    try:
        model = CascadeForestRegressor(
            n_estimators=best_params['n_estimators'],
            n_trees=best_params['n_trees'],
            max_layers=best_params['max_layers'],
            n_bins=best_params['n_bins'],
            bin_subsample=best_params['bin_subsample'],
            random_state=42,
            n_jobs=-1,
            partial_mode='true'
        )
        model.fit(X_train_lasso_selected.values, y_train_scaled)
        logging.info("模型训练完成。")
    except Exception as e:
        logging.error(f"模型训练失败: {e}")
        raise

    # 模型预测与评估
    logging.info("\n进行模型预测...")
    try:
        y_pred_scaled = model.predict(X_test_lasso_selected.values).flatten()  # 预测并转换为一维数组
    except Exception as e:
        logging.error(f"模型预测失败: {e}")
        raise

    # 将预测值反标准化
    try:
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        logging.info("预测值已反标准化。")
    except Exception as e:
        logging.error(f"预测值反标准化失败: {e}")
        raise

    # 评估模型
    try:
        # 对齐 y_test 与 X_test_transformed 的索引
        y_test_aligned = y_test.loc[X_test_transformed.index]
        r2 = r2_score(y_test_aligned, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_aligned, y_pred))
        mae = mean_absolute_error(y_test_aligned, y_pred)

        logging.info("\n评估指标:")
        logging.info(f"R²: {r2:.4f}")
        logging.info(f"RMSE: {rmse:.4f}")
        logging.info(f"MAE: {mae:.4f}")
    except Exception as e:
        logging.error(f"模型评估失败: {e}")
        raise

    # 打印选择的特征
    logging.info("\n选择的特征列表:")
    for feature in selected_features:
        logging.info(feature)

    # 打印未选择的特征
    logging.info("\n未选择的特征列表:")
    for feature in not_selected_features:
        logging.info(feature)


def forward_feature_selection_single_split_with_optimization(X_train, y_train, base_features=None, max_features=None,
                                                             random_state=42):
    if base_features is None:
        base_features = []

    selected_features = base_features.copy()
    remaining_features = sorted(list(set(X_train.columns) - set(selected_features)))
    feature_addition_order = []
    start_time = time.time()

    # 第一步：将数据集划分为训练集（60%）和临时集（40%）
    X_train, X_temp, y_train, y_temp = train_test_split(X_train, y_train, train_size=0.6, random_state=random_state)

    # 第二步：将临时集划分为验证集（20%）和测试集（20%）
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_state)

    logging.info(f"训练集大小: {X_train.shape[0]}, 验证集大小: {X_val.shape[0]}, 测试集大小: {X_test.shape[0]}")

    # 对 CascadeForestRegressor 进行调参
    best_params = optimize_cascade_forest(X_train.values, y_train)

    # 评估基础特征模型性能作为基线，使用调优后的模型参数
    if selected_features:
        logging.info("评估基础特征模型性能作为基线...")
        try:
            base_model = CascadeForestRegressor(n_estimators=best_params['n_estimators'],
                                                n_trees=best_params['n_trees'],
                                                max_layers=best_params['max_layers'],
                                                n_bins=best_params['n_bins'],
                                                bin_subsample=best_params['bin_subsample'],
                                                random_state=42)
            # 直接使用 .values 转换为 NumPy 数组
            base_model.fit(X_train[selected_features].values, y_train)
            y_base_pred = base_model.predict(X_val[selected_features].values)
            base_r2 = r2_score(y_val, y_base_pred)
            logging.info(f"基础特征模型的验证集 R²: {base_r2:.4f}")
        except Exception as e:
            logging.error(f"基础特征模型评估失败: {e}")
            raise
    else:
        base_r2 = -np.inf
        logging.info("没有指定基础特征，基线 R² 设置为 -inf。")

    best_r2 = base_r2
    improvement = True

    while improvement and (max_features is None or len(selected_features) < max_features):
        improvement = False
        best_feature = None
        best_r2_current = best_r2

        logging.info(f"\n当前已选择特征数量: {len(selected_features)}")
        logging.info(f"剩余可选择特征数: {len(remaining_features)}")

        # 评估所有候选特征
        results = []
        total_features = len(remaining_features)
        for idx, feature in enumerate(remaining_features, 1):
            logging.info(f"正在测试特征 {idx}/{total_features}: {feature}")
            try:
                # 使用调优后的参数训练模型
                model = CascadeForestRegressor(n_estimators=best_params['n_estimators'],
                                               n_trees=best_params['n_trees'],
                                               max_layers=best_params['max_layers'],
                                               n_bins=best_params['n_bins'],
                                               bin_subsample=best_params['bin_subsample'],
                                               random_state=42)
                # 直接使用 .values 转换为 NumPy 数组
                model.fit(X_train[selected_features + [feature]].values, y_train)

                # 验证模型（在验证集上）
                y_val_pred = model.predict(X_val[selected_features + [feature]].values)
                val_r2 = r2_score(y_val, y_val_pred)
                logging.info(f"特征 {feature} 的验证集 R²: {val_r2:.4f}")

                # 在测试集上也进行验证
                y_test_pred = model.predict(X_test[selected_features + [feature]].values)
                test_r2 = r2_score(y_test, y_test_pred)
                logging.info(f"特征 {feature} 的测试集 R²: {test_r2:.4f}")

                # 存储验证集和测试集的结果
                results.append((feature, val_r2, test_r2))
            except Exception as e:
                logging.error(f"特征 {feature} 测试失败: {e}")
                continue

        if not results:
            logging.warning("所有候选特征的评估均失败。停止特征选择。")
            break

        # 找出最佳特征，使用验证集上的 R² 判断
        for feature, val_r2, test_r2 in results:
            if val_r2 > best_r2_current:
                best_r2_current = val_r2
                best_feature = feature

        if best_feature is not None and best_r2_current > best_r2:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            improvement = True
            best_r2 = best_r2_current
            feature_addition_order.append((best_feature, best_r2))
            logging.info(f"特征 {best_feature} 被选择，R² 提升至 {best_r2:.4f}")

            # 进行5折交叉验证
            logging.info(f"对特征 {best_feature} 进行5折交叉验证评估...")
            cross_val_r2 = cross_val_score(model, X_train[selected_features].values, y_train, cv=5, scoring='r2').mean()
            logging.info(f"特征 {best_feature} 的5折交叉验证 R²: {cross_val_r2:.4f}")

            # 添加当前已选择的特征列表日志
            logging.info(f"当前已选择的特征: {selected_features}")
        else:
            logging.info("没有进一步的改进，停止特征选择。")
            break

    end_time = time.time()
    logging.info(
        f"\n前向特征选择完成。共选择了 {len(selected_features)} 个特征（包括基础特征），耗时 {end_time - start_time:.2f} 秒。")

    # 计算未被选择的特征
    not_selected_features = sorted(list(set(X_train.columns) - set(selected_features)))

    logging.info("\n选择的特征顺序及对应的R²:")  # 用于输出每个特征的R²值
    for idx, (feat, r2) in enumerate(feature_addition_order, 1):
        logging.info(f"{idx}. {feat} - R²: {r2:.4f}")

    logging.info("\n未被选择的特征列表:")  # 输出未选择的特征
    for feat in not_selected_features:
        logging.info(feat)

    # 检查是否选择了至少基础特征
    if not selected_features:
        raise ValueError("前向特征选择未选择任何特征。请检查数据和特征工程步骤。")
    final_best_params = optimize_cascade_forest(X_train[selected_features].values, y_train)
    # 在测试集上评估最终模型
    final_model = CascadeForestRegressor(n_estimators=final_best_params['n_estimators'],
                                         n_trees=final_best_params['n_trees'],
                                         max_layers=final_best_params['max_layers'],
                                         n_bins=final_best_params['n_bins'],
                                         bin_subsample=final_best_params['bin_subsample'],
                                         random_state=42)
    final_model.fit(X_train[selected_features].values, y_train)
    y_test_pred_final = final_model.predict(X_test[selected_features].values)

    final_r2 = r2_score(y_test, y_test_pred_final)
    final_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_final))
    final_mae = mean_absolute_error(y_test, y_test_pred_final)

    logging.info(f"最终模型在测试集上的评估：")
    logging.info(f"R²: {final_r2:.4f}")
    logging.info(f"RMSE: {final_rmse:.4f}")
    logging.info(f"MAE: {final_mae:.4f}")

    return selected_features, not_selected_features


def main():
    # 数据加载
    file_path = r'D:\datasets\merge\yearly_merged\merged_2015_2022_combined.csv'

    if not os.path.exists(file_path):
        logging.error(f"文件路径不存在: {file_path}")
        raise FileNotFoundError(f"文件路径不存在: {file_path}")

    logging.info("正在加载数据...")
    try:
        df = pd.read_csv(file_path)

        logging.info("数据加载完成。")
    except Exception as e:
        logging.error(f"数据加载失败: {e}")
        raise

    # 确保 'time' 列存在并格式正确
    if 'time' not in df.columns:
        logging.error("数据中缺少 'time' 列.")
        raise ValueError("数据中缺少 'time' 列.")

    # 确保所有必要的列存在
    REQUIRED_COLUMNS = [
        'lat', 'lon', 'CT', 'dm', 'gp', 'pp', 'sh', 'sp', 'te',
        'NO2', 'CO', 'nt', 'nv', 'XCO2', 'time'
    ]
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        logging.error(f"数据中缺少以下必要列: {missing_columns}")
        raise ValueError(f"数据中缺少以下必要列: {missing_columns}")
    else:
        logging.info("数据包含所有必要列。")

    # 过滤并去重
    logging.info("过滤并去重...")
    try:
        df = df.dropna(subset=['time'])
        df = df[df['time'].astype(str).str.len() == 8]  # 确保 'time' 格式为 YYYYMMDD
        df['date'] = pd.to_datetime(df['time'].astype(str), format='%Y%m%d', errors='coerce')
        df['year'] = df['date'].dt.year
        df = df[df['year'].isin(list(range(2015, 2023)))]

        # 提取月份并筛选1-13月的数据
        df['month'] = df['date'].dt.month
        df = df[df['month'].isin(list(range(1, 13)))]

        # 移除辅助列
        df = df.dropna(subset=['time'])
        df = df.drop(columns=['year', 'month', 'date'])  # 删除辅助列
        logging.info(f"过滤后数据样本数: {df.shape[0]}")
    except Exception as e:
        logging.error(f"数据过滤和去重失败: {e}")
        raise

    # 初始化特征工程器
    logging.info("初始化特征工程器...")
    feature_engineer = FeatureEngineer()

    # 训练模型
    try:
        train_model(df, feature_engineer)
    except Exception as e:
        logging.error(f"训练模型时发生错误: {e}")
        raise


if __name__ == "__main__":
    main()
