import streamlit as st
from joblib import load 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_selection import RFECV, SequentialFeatureSelector
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.linear_model import LinearRegression
import joblib
from time import sleep
import base64
import plotly.graph_objects as go


if 'page_num' not in st.session_state:
    st.session_state.page_num = 1
# 界面1
def page1():
    # 初始化会话状态
    st.title("基于人工智能的水泥厂优化决策模型系统")
    st.markdown('内蒙古科技大学开发')
    st.markdown('使用人工智能的方法，对回转窑、原料磨设备进行优化运行')

    selected_species = st.selectbox('可以优化的设备', ['回转窑', '原料磨'])
    if selected_species in ['回转窑', '原料磨']:
        st.session_state.selected_species = selected_species

    # 添加确认选择按钮
    if st.button('确认选择'):
        if selected_species == '回转窑':
            st.session_state.page_num = 2
            st.rerun()
        elif selected_species == '原料磨':
            st.session_state.page_num = 3
            st.rerun()

    

# 界面2
def page2():
    st.title('回转窑预测模型再训练')
    st.markdown('是否进行回转窑预测模型的再训练？')
    col1, col2, col3 = st.columns(3)
    if col1.button('确认', key='confirm_button'):
        st.session_state.page_num = 4
        st.rerun()

    if col3.button("返回"):
        st.session_state.page_num = 1
        st.rerun()
    if col2.button('取消', key='cancel_button'):
        st.session_state.page_num = 5
        st.rerun()
# 界面3
def page3():
    st.title('原料磨预测模型再训练')
    st.markdown('是否进行原料磨预测模型的再训练？')
    col1, col2, col3 = st.columns(3)

    if col1.button('确认', key='confirm_button'):
        st.session_state.page_num = 6
        st.rerun()

    if col3.button("返回"):
        st.session_state.page_num = 1
        st.rerun()
    
    if col2.button('取消', key='cancel_button'):
        st.session_state.page_num = 7
        st.rerun()
# 界面4
def page4():
        if "page_num" not in st.session_state:
            st.session_state.page_num = 4
        if st.session_state.page_num == 4:
            def train_and_evaluate_model(X_train, X_test, y_train, y_test, model, param_grid):
                grid_search = GridSearchCV(model, param_grid, cv=5, scoring=make_scorer(nrmse))
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_

                # 交叉验证
                scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring=make_scorer(nrmse))
                mean_score = np.mean(scores)

                # 训练最佳模型
                best_model.fit(X_train, y_train)

                # 计算测试分数
                test_score = nrmse(best_model.predict(X_test), y_test)

                return best_model, best_params, mean_score, test_score

            # 定义 NRMSE 评估函数   
            def nrmse(y_true, y_pred):
                return np.sqrt(mean_squared_error(y_true, y_pred)) / np.std(y_true)
            
            # 主函数
            def main():
                st.title("回转窑回归模型的选择与预测")
                # 文件上传部分
                st.sidebar.subheader("第一步：上传数据")
                file = st.sidebar.file_uploader("上传您的 Excel 文件", type=["xlsx"])
                if file is not None:
                    data = pd.read_excel(file)

                    # 获取除第一列外的列名列表
                    column_names = data.columns[1:].tolist()

                    # 显示折线图
                    st.subheader("变量折线图")
                    tabs = st.tabs(column_names)
                    for i, column in enumerate(column_names):
                        with tabs[i]:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=data[data.columns[0]], y=data[column], mode='lines+markers', marker=dict(size=2)))
                            fig.update_layout(title=f"{column}变量折线图", xaxis_title='时间', yaxis_title=column)
                            fig.update_traces(hoverinfo='x+y')  # 鼠标悬停时显示x和y值
                            st.plotly_chart(fig)


                        # 用户选择预测变量和特征变量
                    st.sidebar.subheader("第二步：选择预测变量和自变量")
                    y_columns = st.sidebar.multiselect("预测变量的挑选", data.columns[1:], key="y_columns_multiselect")
                    feature_columns_dict = {}

                    for idx, y_column in enumerate(y_columns):
                        available_features = [col for col in data.columns[1:] if col != y_column]
                        feature_columns_dict[y_column] = st.sidebar.multiselect(f"选择自变量为 '{y_column}'", available_features, key=f"feature_multiselect_{idx}")

                    # 预测变量是否执行特征选择的字典
                    feature_selection_dict = {}
                    for y_column in y_columns:
                        perform_feature_selection = st.sidebar.selectbox(f"是否对'{y_column}'执行特征选择", ['是', '否'], key=f"feature_selection_{y_column}")
                        feature_selection_dict[y_column] = perform_feature_selection

                    if st.sidebar.button("确认"):
                        for y_column in y_columns:
                            perform_feature_selection = feature_selection_dict[y_column]
                            feature_columns = feature_columns_dict[y_column]
                            if perform_feature_selection == "是":
                                y = data[y_column]
                                X = data[feature_columns]

                                # 使用 RFECV 进行特征选择
                                st.sidebar.write(f"正在对'{y_column}'执行特征选择...")
                                estimator = RandomForestRegressor()
                                selector_rfecv = RFECV(estimator, step=1, cv=5)
                                X_rfecv = selector_rfecv.fit_transform(X, y)
                                # 使用 SFS 进行特征选择
                                selector_sfs = SequentialFeatureSelector(estimator, direction='forward', cv=5)
                                selector_sfs.fit(X_rfecv, y)
                                best_num_features = selector_sfs.n_features_to_select_
                                # 获取选定的特征
                                selected_features = X.columns[selector_rfecv.support_][selector_sfs.get_support()]

                                st.write(f"'{y_column}'特征选择完成，选定的特征：{', '.join(selected_features)}")
                                # 用选定的特征进行模型的选择与训练
                                run_model_selection(X[selected_features].values, y, best_num_features, y_column)

                            else:
                                best_num_features = len(feature_columns)

                                # 用所有特征进行模型的选择与训练
                                st.write(f"不执行特征选择，使用所有特征进行模型训练，预测变量： {y_column}...")
                                run_model_selection(data[feature_columns].values, data[y_column], best_num_features, y_column)

            # 运行模型选择
            def run_model_selection(X, y, best_num_features, y_column):
                # 训练测试集拆分
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                models = {
                    "线性回归": LinearRegression(),
                    "KNN 回归": KNeighborsRegressor(),
                    "MLP 回归": MLPRegressor(max_iter=1000),
                    "随机森林回归": RandomForestRegressor(),
                    "梯度提升回归": GradientBoostingRegressor(),
                    "LGBM 回归": LGBMRegressor(force_col_wise=True),
                    "XGBoost 回归": XGBRegressor(),
                    "CatBoost 回归": CatBoostRegressor(silent=True)
                }

                param_grids = {  # 为每个模型定义超参数网格
                    "线性回归": {},
                    "KNN 回归": {'n_neighbors': [3, 5, 7], 'leaf_size': [20, 30, 50]},
                    "MLP 回归": {'hidden_layer_sizes': [(20,), (50,), (50, 50)], 'alpha': [0.0001, 0.001, 0.01]},
                    "随机森林回归": {'n_estimators': [30, 50, 100], 'max_depth': [None, 10, 20]},
                    "梯度提升回归": {'n_estimators': [30, 50, 100], 'max_depth': [5, 7, 10]},
                    "LGBM 回归": {'n_estimators': [30, 50, 70], 'max_depth': [10, 15, 20], 'num_leaves': [5, 10, 15]},
                    "XGBoost 回归": {'n_estimators': [30, 50, 70], 'max_depth': [5, 7, 10]},
                    "CatBoost 回归": {'n_estimators': [30, 50, 70], 'max_depth': [5, 7, 10]}
                }

                best_models = {}
                best_params = {}
                scores = {}
                test_scores = {}

                # 显示加载指示器
                with st.spinner('正在训练模型...'):
                    for name, model in models.items():
                        sleep(1)  # 模拟模型训练的时间，可删除
                        message = st.sidebar.empty()
                        message.write(f"正在训练 {name}...")
                        best_model, best_param, score, test_score = train_and_evaluate_model(X_train, X_test, y_train, y_test, model, param_grids[name])
                        best_models[name] = best_model
                        best_params[name] = best_param
                        scores[name] = score
                        test_scores[name] = test_score
                        message.write(f"训练完成 {name}...")

                # 获取最佳模型名称
                best_model_name = min(scores, key=scores.get)
                st.write(f"最佳模型：{best_model_name}")
                st.write(f"最佳模型的 NRMSE 分数：{scores[best_model_name]}")
                
                # 创建临时文件路径并保存最佳模型
                best_model_name = min(scores, key=scores.get)
                best_model = best_models[best_model_name]
                best_model_filename = f'best_model_{y_column}_{best_num_features}.pkl'
                joblib.dump(best_model, best_model_filename)

                # 使用 base64 编码模型文件并生成下载链接
                with open(best_model_filename, 'rb') as f:
                    model_content = f.read()
                    model_base64 = base64.b64encode(model_content).decode('utf-8')
                    download_link = f'<a href="data:application/octet-stream;base64,{model_base64}" download="{best_model_filename}">点击下载最佳模型</a>'
                    st.markdown(download_link, unsafe_allow_html=True)
            col1, col2 = st.sidebar.columns(2)
            if col1.button("返回", key="1"):
                st.session_state.page_num = 2
                st.rerun() 
            if col2.button("继续", key="0"):
                st.session_state.page_num = 5
                st.rerun() 
            # 运行
            if __name__ == "__main__":
                main()

# 界面5
def page5():
    if "page_num" not in st.session_state:
        st.session_state.page_num = 5
    if st.session_state.page_num == 5:
        def load_model(model_file):
            model = load(model_file)
            return model

        # 预测函数
        def predict_data(data, models, model_indices):
            predictions = []
            for model_index, model in zip(model_indices, models):
                if model_index == 1:  
                    prediction = model.predict(data[:, [1, 2]]) 
                elif model_index == 2:  
                    prediction = model.predict(data[:, [0, 2]]) 
                elif model_index == 3:  
                    prediction = model.predict(data[:, [1, 2]]) 
                elif model_index == 4:  
                    prediction = model.predict(data[:, [2, 3]])  
                elif model_index == 5:  
                    prediction = model.predict(data[:, [0, 1, 2, 3]])
                elif model_index == 6:  
                    prediction = model.predict(data[:, [0, 1, 2, 3]]) 
                else:
                    raise ValueError("Invalid model index:", model_index)
                predictions.extend(prediction.flatten())  
            return np.array(predictions)

        # 目标函数
        def objective_function(predicted, W_bounds, W_min_boolean, W_max_boolean, W_monotony, W_target, sign, minimum, maximum, mean, target, std):
            predicted = np.array(predicted)  
            sign = np.array(sign) 
            W_monotony = np.array(W_monotony) 
            W_bounds = np.array(W_bounds)  
            W_min_boolean = np.array(W_min_boolean)  
            W_max_boolean = np.array(W_max_boolean)  
            W_target = np.array(W_target) 
            minimum = np.array(minimum) 
            maximum = np.array(maximum)  
            mean = np.array(mean)  
            std = np.array(std) 
            
            term1 = np.sum(W_bounds * (
                W_min_boolean * ((predicted - minimum) / std) ** 2 * (predicted < minimum) + 
                W_max_boolean * ((predicted - maximum) / std) ** 2 * (predicted > maximum)
            ))
            term2 = np.sum(sign * W_monotony * (predicted - mean) / std + 100000)
            term3 = np.sum(W_target * ((predicted - target) / std) ** 2)
            
            return term1 + term2 + term3

        # 初始化种群
        class Individual:
            def __init__(self, genes):
                self.genes = genes
                self.fitness = None 
                
        def initialize_population(population_size, bounds):
            population = []
            for _ in range(population_size):
                genes = [np.random.normal((low + high) / 2, (high - low) / 6) for low, high in bounds]
                individual = Individual(genes)
                population.append(individual)
            return population

        # 变异、交叉和选择操作
        def evolve(population, objective_function, crossover_rate, mutation_factor, bounds):
            new_population = []
            total_fitness = sum(individual.fitness for individual in population)
            probabilities = [individual.fitness / total_fitness for individual in population]

            for _ in range(len(population)):
                selected_index = np.random.choice(len(population), p=probabilities)
                selected_individual = population[selected_index]

                a, b, c, = np.random.choice(population, 3, replace=False)
                mutated_individual = np.clip(np.array(a.genes) + mutation_factor * (np.array(b.genes) - np.array(c.genes)), bounds[:, 0], bounds[:, 1])
                mask = np.random.rand(len(selected_individual.genes)) < crossover_rate
                crossed_individual = np.where(mask, mutated_individual, selected_individual.genes)

                new_population.append(Individual(crossed_individual))

            return new_population

        # 主函数
        def main():
            st.title("回转窑操作变量的最优值推荐")
            # 用户上传 Excel 数据
            uploaded_file = st.file_uploader("用户上传数据", type=["xls", "xlsx"])
            if uploaded_file is not None:
                df = pd.read_excel(uploaded_file, sheet_name='Sheet1')
                mean_values = df.iloc[:, 1:].mean()
                std_values = df.iloc[:, 1:].std()
                mean = mean_values.values
                std = std_values.values
                # 获取除第一列外的列名列表
                column_names = df.columns[1:].tolist()
                # 显示折线图
                st.subheader("变量折线图")
                tabs = st.tabs(column_names)
                for i, column in enumerate(df.columns[1:]):
                    with tabs[i]:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df[df.columns[0]], y=df[column], mode='lines+markers', marker=dict(size=2)))  # 修改 marker 的 size
                        fig.update_layout(title=f"{column}变量折线图", xaxis_title='时间', yaxis_title=column)
                        fig.update_traces(hoverinfo='x+y')  
                        st.plotly_chart(fig)
                

                st.sidebar.subheader("操作变量边界设定")
                column_names_fixed = ["窑尾煤粉投入量", "窑头煤粉投入量", "转速", "一次风量", "均化库喂料量"]
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    min_value_1 = st.number_input("窑尾煤粉投入量min", value=1.737, format="%.4f", key="min_value_1")
                    min_value_2 = st.number_input("窑头煤粉投入量min", value=4.824, format="%.4f", key="min_value_2")
                    min_value_3 = st.number_input("转速min", value=3.69, format="%.4f", key="min_value_3")
                    min_value_4 = st.number_input("一次风量min", value=25363.629, format="%.4f", key="min_value_4")
                    min_value_5 = st.number_input("均化库喂料量min", value=180.00, format="%.4f", key="min_value_5")

                with col2:
                    max_value_1 = st.number_input("窑尾煤粉投入量max", value=2.12, format="%.4f", key="max_value_1")
                    max_value_2 = st.number_input("窑头煤粉投入量max", value=5.896, format="%.4f", key="max_value_2")
                    max_value_3 = st.number_input("转速max", value=4.62, format="%.4f", key="max_value_3")
                    max_value_4 = st.number_input("一次风量max", value=29856.35, format="%.4f", key="max_value_4")
                    max_value_5 = st.number_input("均化库喂料量max", value=195.00, format="%.4f", key="max_value_5")

                    bounds_min = [min_value_1, min_value_2, min_value_3, min_value_4, min_value_5]
                    bounds_max = [max_value_1, max_value_2, max_value_3, max_value_4, max_value_5]
                    bounds = np.array(list(zip(bounds_min, bounds_max)))


                    st.sidebar.subheader("非操作变量边界设定")
                    col1, col2 = st.sidebar.columns(2)
                with col1:
                    min_value_a = st.number_input("回转窑电流min", value=486.64, format="%.4f", key="min_value_a")
                    min_value_b = st.number_input("窑头气体温度min", value=980.65, format="%.4f", key="min_value_b")
                    min_value_c = st.number_input("分解炉温度min", value=812.3, format="%.4f", key="min_value_c")
                    min_value_d = st.number_input("一级筒温度min", value=449.77, format="%.4f", key="min_value_d")
                    min_value_e = st.number_input("二级筒温度min", value=727.82, format="%.4f", key="min_value_e")
                    min_value_f = st.number_input("三级筒温度min", value=212.13, format="%.4f", key="min_value_f")
                    min_value_g = 0
                with col2:
                    max_value_a = st.number_input("回转窑电流max", value=594.781, format="%.4f", key="max_value_a")
                    max_value_b = st.number_input("窑头气体温度max", value=1174.47, format="%.4f", key="max_value_b")
                    max_value_c = st.number_input("分解炉温度max", value=950.6, format="%.4f", key="max_value_c")
                    max_value_d = st.number_input("一级筒温度max", value=549.747, format="%.4f", key="max_value_d")
                    max_value_e = st.number_input("二级筒温度max", value=889.559, format="%.4f", key="max_value_e")
                    max_value_f = st.number_input("三级筒温度max", value=259.27, format="%.4f", key="max_value_f")
                    max_value_g = 0 

                    minimum= [min_value_a, min_value_b, min_value_c, min_value_d, min_value_e, min_value_f, min_value_g]
                    maximum= [max_value_a, max_value_b, max_value_c, max_value_d, max_value_e, max_value_f, max_value_g]

                st.sidebar.subheader("变量权重的确定")
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    weight_options = ["极其重要", "非常重要", "重要", "中等", "一般", "不太重要", "可忽略的"]
                    W_bounds = st.selectbox("回转窑电流权重", weight_options, index=0, key="W_bounds_1")
                    W_bounds = [1000000, 100000, 10000, 1000, 100, 10, 1][weight_options.index(W_bounds)]

                    W_bounds_2 = st.selectbox("窑头气体温度权重", weight_options, index=1, key="W_bounds_2")
                    W_bounds_2 = [1000000, 100000, 10000, 1000, 100, 10, 1][weight_options.index(W_bounds_2)]

                    W_bounds_3 = st.selectbox("分解炉温度权重", weight_options, index=2, key="W_bounds_3")
                    W_bounds_3 = [1000000, 100000, 10000, 1000, 100, 10, 1][weight_options.index(W_bounds_3)]
                with col2:
                    W_bounds_4 = st.selectbox("一级筒温度权重", weight_options, index=3, key="W_bounds_4")
                    W_bounds_4 = [1000000, 100000, 10000, 1000, 100, 10, 1][weight_options.index(W_bounds_4)]

                    W_bounds_5 = st.selectbox("二级筒温度权重", weight_options, index=4, key="W_bounds_5")
                    W_bounds_5 = [1000000, 100000, 10000, 1000, 100, 10, 1][weight_options.index(W_bounds_5)]

                    W_bounds_6 = st.selectbox("三级筒温度权重", weight_options, index=5, key="W_bounds_6")
                    W_bounds_6 = [1000000, 100000, 10000, 1000, 100, 10, 1][weight_options.index(W_bounds_6)]

                    W_bounds_7 = 0

                    W_bounds = [W_bounds, W_bounds_2, W_bounds_3, W_bounds_4, W_bounds_5, W_bounds_6, W_bounds_7]
                    
                    W_min_boolean = np.array([1, 1, 1, 1, 1, 1, 0])
                    W_max_boolean = np.array([1, 1, 1, 1, 1, 1, 0])
                    
                    st.sidebar.subheader("变量单调性权重的确定")
                    col1, col2 = st.sidebar.columns(2)
                    with col1:
                        monotony_options = ["重要", "中等", "一般", "不太重要", "可忽略的"]
                        W_monotony_1 = st.selectbox("回转窑电流权重", monotony_options, index=0, key="W_monotony_1")
                        W_monotony_1 = [10000, 1000, 100, 10, 1][monotony_options.index(W_monotony_1)]
                        
                        W_monotony_2 = st.selectbox("窑头气体温度权重", monotony_options, index=0, key="W_monotony_2")
                        W_monotony_2 = [10000, 1000, 100, 10, 1][monotony_options.index(W_monotony_2)]
                        
                        W_monotony_3 = st.selectbox("分解炉温度权重", monotony_options, index=0, key="W_monotony_3")
                        W_monotony_3 = [10000, 1000, 100, 10, 1][monotony_options.index(W_monotony_3)]
                    with col2:
                        W_monotony_4 = st.selectbox("一级筒温度权重", monotony_options, index=0, key="W_monotony_4")
                        W_monotony_4 = [10000, 1000, 100, 10, 1][monotony_options.index(W_monotony_4)]
                        
                        W_monotony_5 = st.selectbox("二级筒温度权重", monotony_options, index=0, key="W_monotony_5")
                        W_monotony_5 = [10000, 1000, 100, 10, 1][monotony_options.index(W_monotony_5)]
                        
                        W_monotony_6 = st.selectbox("三级筒温度权重", monotony_options, index=0, key="W_monotony_6")
                        W_monotony_6 = [10000, 1000, 100, 10, 1][monotony_options.index(W_monotony_6)]
                        W_monotony_7 = 0
                    W_monotony = [W_monotony_1, W_monotony_2, W_monotony_3, W_monotony_4, W_monotony_5, W_monotony_6, W_monotony_7]
                    
                    st.sidebar.subheader("变量单调性确定")
                    col1, col2, col3 = st.sidebar.columns(3)

                    with col1:
                        monotonicity_options = {"单调递增": -1, "单调递减": 1}
                        sign_1 = st.selectbox("回转窑", list(monotonicity_options.keys()), index=0, key="sign_1")
                        sign_1 = monotonicity_options[sign_1]

                        sign_2 = st.selectbox("窑头气体温度", list(monotonicity_options.keys()), index=1, key="sign_2")
                        sign_2 = monotonicity_options[sign_2]
                    with col2:
                        sign_3 = st.selectbox("分解炉温度", list(monotonicity_options.keys()), index=0, key="sign_3")
                        sign_3 = monotonicity_options[sign_3]
                        
                        sign_4 = st.selectbox("一级筒温度", list(monotonicity_options.keys()), index=0, key="sign_4")
                        sign_4 = monotonicity_options[sign_4]

                    with col3:
                        sign_5 = st.selectbox("二级筒温度", list(monotonicity_options.keys()), index=0, key="sign_5")
                        sign_5 = monotonicity_options[sign_5]
                        
                        sign_6 = st.selectbox("三级筒温度", list(monotonicity_options.keys()), index=1, key="sign_6")
                        sign_6 = monotonicity_options[sign_6]
                        sign_7 = 0

                    sign = [sign_1, sign_2, sign_3, sign_4, sign_5, sign_6, sign_7]
                    
                    st.sidebar.subheader("目标值权重的确定")
                    W_target = st.sidebar.text_input("均化库喂料量权重", value="10")
                    W_target = [0, 0, 0, 0, 0, 0, float(W_target)]
                    st.sidebar.subheader("目标值")
                    target = st.sidebar.text_input("均化库喂料量", value="185")
                    target = [0, 0, 0, 0, 0, 0, float(target.strip().split(",")[-1])]
                    st.sidebar.subheader("调整参数")
                    population_size = st.sidebar.number_input("种群大小", min_value=5, max_value=1000, value=20)
                    max_generations = st.sidebar.number_input("最大迭代次数", min_value=100, max_value=2000, value=200)
                    
                    models = None
                    model_indices = None
                    st.sidebar.subheader("上传模型文件")
                    model_files = st.sidebar.file_uploader("**回转窑电流** > 窑头气体温度 > **分解炉温度** > 一级筒温度 > **二级筒温度** > 三级筒温度", accept_multiple_files=True, type=["pkl"])
                    if len(model_files) > 0:  # 检查文件列表长度是否大于0
                        models = [load_model(file) for file in model_files]
                        model_indices = [1, 2, 3, 4, 5, 6] 


                if st.button("Confirm"):
                    if model_files is not None:
                        crossover_rate = 0.8
                        mutation_factor = 0.5
                        population = initialize_population(population_size, bounds)
                # 迭代优化
                    best_individual = None
                    best_fitness = float('inf')
                    with st.spinner('正在进行差分进化...'):
                        for generation in range(max_generations):
                            current_best_fitness = float('inf')
                            for individual in population:
                                # 提取第6组数据的基因值
                                gene_value_to_use = individual.genes[4]  # 第6组数据的索引为5（从0开始）
                                # 构建包含第6组数据基因值的数据点
                                genes_with_sixth_value = individual.genes.copy()
                                genes_with_sixth_value[4] = gene_value_to_use
                                # 使用预测函数进行预测
                                predictions = predict_data(np.array(genes_with_sixth_value).reshape(1, -1), models, model_indices)
                                predictions = np.append(predictions, gene_value_to_use)
                                # 计算适应度值
                                fitness = objective_function(predictions, W_bounds, W_min_boolean, W_max_boolean, W_monotony, W_target, sign, minimum, maximum, mean, target, std)
                                individual.fitness = fitness
                                print("Genes:", individual.genes)
                                print("Predictions:", predictions)
                                print("Fitness:", fitness)
                                # 更新最佳个体和最佳目标函数值
                                if fitness < current_best_fitness:
                                    current_best_fitness = fitness
                                    current_best_individual = individual

                            # 更新全局最佳个体和最佳目标函数值
                            if current_best_fitness < best_fitness:
                                best_fitness = current_best_fitness
                                best_individual = current_best_individual

                            # 进化操作
                            population = evolve(population, objective_function, crossover_rate, mutation_factor, bounds)
                        # 输出最终的最佳个体和最佳目标函数值
                        st.subheader("Best Individual:")
                        if best_individual is not None:
                            st.write("Recommended Operation Variables:")
                            for i, variable_name in enumerate(column_names_fixed):
                                st.write(f"{variable_name}: {best_individual.genes[i]:.4f}")
                            st.write("Minimum Fitness of the Model: {:.4f}".format(best_fitness))
                            st.success("Best individual and minimum fitness found successfully!")
                        else:
                            st.write("No best individual found.")
                            st.error("Failed to find the best individual.")
            col1, col2 = st.columns(2)

            if col1.button("返回第一步"):
                st.session_state.page_num = 1
                st.rerun()

            if col2.button("返回"):
                st.session_state.page_num = 2
                st.rerun()

        if __name__ == "__main__":
            main()


def page6():
    if "page_num" not in st.session_state:
        st.session_state.page_num = 6
    if st.session_state.page_num == 6:
        # 定义内部函数
        def train_and_evaluate_model(X_train, X_test, y_train, y_test, model, param_grid):
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring=make_scorer(nrmse))
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_

            # 交叉验证
            scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring=make_scorer(nrmse))
            mean_score = np.mean(scores)

            # 训练最佳模型
            best_model.fit(X_train, y_train)

            # 计算测试分数
            test_score = nrmse(best_model.predict(X_test), y_test)

            return best_model, best_params, mean_score, test_score

        # 定义 NRMSE 评估函数   
        def nrmse(y_true, y_pred):
            return np.sqrt(mean_squared_error(y_true, y_pred)) / np.std(y_true)

        # 主函数
        def main():
            st.title("原料磨回归模型的选择与预测")

            # 文件上传部分
            st.sidebar.subheader("第一步：上传数据")
            file = st.sidebar.file_uploader("上传您的 Excel 文件", type=["xlsx"])
            if file is not None:
                data = pd.read_excel(file)

                # 获取除第一列外的列名列表
                column_names = data.columns[1:].tolist()

                # 显示折线图
                st.subheader("变量折线图")
                tabs = st.tabs(column_names)
                for i, column in enumerate(column_names):
                    with tabs[i]:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=data[data.columns[0]], y=data[column], mode='lines+markers', marker=dict(size=2)))
                        fig.update_layout(title=f"{column}变量折线图", xaxis_title='时间', yaxis_title=column)
                        fig.update_traces(hoverinfo='x+y')  # 鼠标悬停时显示x和y值
                        st.plotly_chart(fig)

                # 用户选择预测变量和特征变量
                st.sidebar.subheader("第二步：选择预测变量和自变量")
                y_columns = st.sidebar.multiselect("预测变量的挑选", data.columns[1:], key="y_columns_multiselect")
                feature_columns_dict = {}

                for idx, y_column in enumerate(y_columns):
                    available_features = [col for col in data.columns[1:] if col != y_column]
                    feature_columns_dict[y_column] = st.sidebar.multiselect(f"选择自变量为 '{y_column}'", available_features, key=f"feature_multiselect_{idx}")

                # 预测变量是否执行特征选择的字典
                feature_selection_dict = {}
                for y_column in y_columns:
                    perform_feature_selection = st.sidebar.selectbox(f"是否对'{y_column}'执行特征选择", ['是', '否'], key=f"feature_selection_{y_column}")
                    feature_selection_dict[y_column] = perform_feature_selection

                if st.sidebar.button("确认"):
                    for y_column in y_columns:
                        perform_feature_selection = feature_selection_dict[y_column]
                        feature_columns = feature_columns_dict[y_column]
                        if perform_feature_selection == "是":
                            y = data[y_column]
                            X = data[feature_columns]

                            # 使用 RFECV 进行特征选择
                            st.sidebar.write(f"正在对'{y_column}'执行特征选择...")
                            estimator = RandomForestRegressor()
                            selector_rfecv = RFECV(estimator, step=1, cv=5)
                            X_rfecv = selector_rfecv.fit_transform(X, y)
                            # 使用 SFS 进行特征选择
                            selector_sfs = SequentialFeatureSelector(estimator, direction='forward', cv=5)
                            selector_sfs.fit(X_rfecv, y)
                            best_num_features = selector_sfs.n_features_to_select_
                            # 获取选定的特征
                            selected_features = X.columns[selector_rfecv.support_][selector_sfs.get_support()]

                            st.write(f"'{y_column}'特征选择完成，选定的特征：{', '.join(selected_features)}")
                            # 用选定的特征进行模型的选择与训练
                            run_model_selection(X[selected_features].values, y, best_num_features, y_column)

                        else:
                            best_num_features = len(feature_columns)

                            # 用所有特征进行模型的选择与训练
                            st.write(f"不执行特征选择，使用所有特征进行模型训练，预测变量： {y_column}...")
                            run_model_selection(data[feature_columns].values, data[y_column], best_num_features, y_column)
        # 运行模型选择
        def run_model_selection(X, y, best_num_features, y_column):
            # 训练测试集拆分
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            models = {
                "线性回归": LinearRegression(),
                "KNN 回归": KNeighborsRegressor(),
                "MLP 回归": MLPRegressor(max_iter=1000),
                "随机森林回归": RandomForestRegressor(),
                "梯度提升回归": GradientBoostingRegressor(),
                "LGBM 回归": LGBMRegressor(force_col_wise=True),
                "XGBoost 回归": XGBRegressor(),
                "CatBoost 回归": CatBoostRegressor(silent=True)
            }

            param_grids = {  # 为每个模型定义超参数网格
                "线性回归": {},
                "KNN 回归": {'n_neighbors': [3, 5, 7], 'leaf_size': [20, 30, 50]},
                "MLP 回归": {'hidden_layer_sizes': [(20,), (50,), (50, 50)], 'alpha': [0.0001, 0.001, 0.01]},
                "随机森林回归": {'n_estimators': [30, 50, 100], 'max_depth': [None, 10, 20]},
                "梯度提升回归": {'n_estimators': [30, 50, 100], 'max_depth': [5, 7, 10]},
                "LGBM 回归": {'n_estimators': [30, 50, 70], 'max_depth': [10, 15, 20], 'num_leaves': [5, 10, 15]},
                "XGBoost 回归": {'n_estimators': [30, 50, 70], 'max_depth': [5, 7, 10]},
                "CatBoost 回归": {'n_estimators': [30, 50, 70], 'max_depth': [5, 7, 10]}
            }

            best_models = {}
            best_params = {}
            scores = {}
            test_scores = {}

            # 显示加载指示器
            with st.spinner('正在训练模型...'):
                for name, model in models.items():
                    sleep(1)  # 模拟模型训练的时间，可删除
                    message = st.sidebar.empty()
                    message.write(f"正在训练 {name}...")
                    best_model, best_param, score, test_score = train_and_evaluate_model(X_train, X_test, y_train, y_test, model, param_grids[name])
                    best_models[name] = best_model
                    best_params[name] = best_param
                    scores[name] = score
                    test_scores[name] = test_score
                    message.write(f"训练完成 {name}...")

            # 获取最佳模型名称
            best_model_name = min(scores, key=scores.get)
            st.write(f"最佳模型：{best_model_name}")
            st.write(f"最佳模型的 NRMSE 分数：{scores[best_model_name]}")
            
            # 创建临时文件路径并保存最佳模型
            best_model_name = min(scores, key=scores.get)
            best_model = best_models[best_model_name]
            best_model_filename = f'best_model_{y_column}_{best_num_features}.pkl'
            joblib.dump(best_model, best_model_filename)

            # 使用 base64 编码模型文件并生成下载链接
            with open(best_model_filename, 'rb') as f:
                model_content = f.read()
                model_base64 = base64.b64encode(model_content).decode('utf-8')
                download_link = f'<a href="data:application/octet-stream;base64,{model_base64}" download="{best_model_filename}">点击下载最佳模型</a>'
                st.markdown(download_link, unsafe_allow_html=True)

            st.write("---")  # 添加分隔符
            
        # 使用 st.columns 布局方法将按钮放置在一行上
        col1, col2 = st.sidebar.columns(2)
        if col1.button("返回", key="1"):
            st.session_state.page_num = 3
            st.rerun() 
        if col2.button("继续", key="0"):
            st.session_state.page_num = 7
            st.rerun()

            
        # 运行主函数
        if __name__ == "__main__":
            main()



        
  
        
def page7():
    if "page_num" not in st.session_state:
        st.session_state.page_num = 7
    if st.session_state.page_num == 7:
        def load_model(model_file):
            model = load(model_file)
            return model

        # 预测函数
        def predict_data(data, models, model_indices):
            predictions = []
            for model_index, model in zip(model_indices, models):
                if model_index == 1:  
                    prediction = model.predict(data[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]) 
                elif model_index == 2:  
                    prediction = model.predict(data[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12]]) 
                elif model_index == 3:  
                    prediction = model.predict(data[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12]]) 
                elif model_index == 4:  
                    prediction = model.predict(data[:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]])  
                else:
                    raise ValueError("Invalid model index:", model_index)
                predictions.append(prediction.flatten())  # 将预测结果添加到列表中
            return np.array(predictions)


        # 目标函数
        def objective_function(predicted, W_bounds, W_min_boolean, W_max_boolean, W_monotony, W_target, sign, minimum, maximum, mean, target, std):
            predicted = np.array(predicted)  
            sign = np.array(sign) 
            W_monotony = np.array(W_monotony) 
            W_bounds = np.array(W_bounds)  
            W_min_boolean = np.array(W_min_boolean)  
            W_max_boolean = np.array(W_max_boolean)  
            W_target = np.array(W_target) 
            minimum = np.array(minimum) 
            maximum = np.array(maximum)  
            mean = np.array(mean)  
            std = np.array(std) 
            
            term1 = np.sum(W_bounds * (
                W_min_boolean * ((predicted - minimum) / std) ** 2 * (predicted < minimum) + 
                W_max_boolean * ((predicted - maximum) / std) ** 2 * (predicted > maximum)
            ))
            term2 = np.sum(sign * W_monotony * (predicted - mean) / std + 100000)
            term3 = np.sum(W_target * ((predicted - target) / std) ** 2)
            
            return term1 + term2 + term3

        # 初始化种群
        class Individual:
            def __init__(self, genes):
                self.genes = genes
                self.fitness = None 
                
        def initialize_population(population_size, bounds):
            population = []
            for _ in range(population_size):
                genes = [np.random.normal((low + high) / 2, (high - low) / 6) for low, high in bounds]
                individual = Individual(genes)
                population.append(individual)
            return population

        # 变异、交叉和选择操作
        def evolve(population, objective_function, crossover_rate, mutation_factor, bounds):
            new_population = []
            total_fitness = sum(individual.fitness for individual in population)
            probabilities = [individual.fitness / total_fitness for individual in population]

            for _ in range(len(population)):
                selected_index = np.random.choice(len(population), p=probabilities)
                selected_individual = population[selected_index]

                a, b, c, = np.random.choice(population, 3, replace=False)
                mutated_individual = np.clip(np.array(a.genes) + mutation_factor * (np.array(b.genes) - np.array(c.genes)), bounds[:, 0], bounds[:, 1])
                mask = np.random.rand(len(selected_individual.genes)) < crossover_rate
                crossed_individual = np.where(mask, mutated_individual, selected_individual.genes)

                new_population.append(Individual(crossed_individual))

            return new_population

        # 主函数
        def main():
            st.title("原料磨操作变量的最优值推荐")
            # 用户上传 Excel 数据
            uploaded_file = st.file_uploader("用户上传数据", type=["xls", "xlsx"])
            if uploaded_file is not None:
                df = pd.read_excel(uploaded_file, sheet_name='Sheet1')
                mean_values = df.iloc[:, 1:].mean()
                std_values = df.iloc[:, 1:].std()
                mean = mean_values.values
                std = std_values.values
                # 获取除第一列外的列名列表
                column_names = df.columns[1:].tolist()
                # 显示折线图
                st.subheader("变量折线图")
                tabs = st.tabs(column_names)
                for i, column in enumerate(df.columns[1:]):
                    with tabs[i]:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df[df.columns[0]], y=df[column], mode='lines+markers', marker=dict(size=2)))  # 修改 marker 的 size
                        fig.update_layout(title=f"{column}变量折线图", xaxis_title='时间', yaxis_title=column)
                        fig.update_traces(hoverinfo='x+y')  
                        st.plotly_chart(fig)
                

                st.sidebar.subheader("操作变量边界设定")
                column_names_fixed = ["配料量", "钢渣占比", "硅质原料占比", "粉煤灰占比", "天然砂占比","转炉渣占比", "煤占比", "湿煤渣占比", "原料磨细度设定", "磨头热风进口压力","磨尾热风进口压力", "窑头煤粉", "窑尾煤粉"]
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    min_value_1 = st.number_input("配料量min", value=52.164, format="%.4f", key="min_value_1")
                    min_value_2 = st.number_input("钢渣占比min", value=0.000, format="%.4f", key="min_value_2")
                    min_value_3 = st.number_input("硅质原料占比min", value=0.07875, format="%.4f", key="min_value_3")
                    min_value_4 = st.number_input("粉煤灰占比min", value=0.10199, format="%.4f", key="min_value_4")
                    min_value_5 = st.number_input("天然砂占比min", value=0.2561, format="%.4f", key="min_value_5")
                    min_value_6 = st.number_input("转炉渣占比min", value=0.0775, format="%.4f", key="min_value_6")
                    min_value_7 = st.number_input("煤占比min", value=0.000469, format="%.4f", key="min_value_7")
                    min_value_8 = st.number_input("湿煤渣占比min", value=0.1859, format="%.4f", key="min_value_8")
                    min_value_9 = st.number_input("原料磨细度设定min", value=12.000, format="%.4f", key="min_value_9")
                    min_value_10 = st.number_input("磨头热风进口压力min", value=-1315.35, format="%.4f", key="min_value_10")
                    min_value_11 = st.number_input("磨尾热风进口压力min", value=-1080.95, format="%.4f", key="min_value_11")
                    min_value_12 = st.number_input("窑头煤粉min", value=3.752, format="%.4f", key="min_value_12")
                    min_value_13 = st.number_input("窑尾煤粉min", value=1.351, format="%.4f", key="min_value_13")

                with col2:
                    max_value_1 = st.number_input("配料量max", value=96.876, format="%.4f", key="max_value_1")
                    max_value_2 = st.number_input("钢渣占比max", value=0.00, format="%.4f", key="max_value_2")
                    max_value_3 = st.number_input("硅质原料占比max", value=0.14625, format="%.4f", key="max_value_3")
                    max_value_4 = st.number_input("粉煤灰占比max", value=9.1894, format="%.4f", key="max_value_4")
                    max_value_5 = st.number_input("天然砂占比max", value=0.4755, format="%.4f", key="max_value_5")
                    max_value_6 = st.number_input("转炉渣占比max", value=0.1439, format="%.4f", key="max_value_6")
                    max_value_7 = st.number_input("煤占比max", value=0.0008723, format="%.4f", key="max_value_7")
                    max_value_8 = st.number_input("湿煤渣占比max", value=0.3453, format="%.4f", key="max_value_8")
                    max_value_9 = st.number_input("原料磨细度设定max", value=14.00, format="%.4f", key="max_value_9")
                    max_value_10 = st.number_input("磨头热风进口压力max", value=-708.267, format="%.4f", key="max_value_10")
                    max_value_11 = st.number_input("磨尾热风进口压力max", value=-582.05, format="%.4f", key="max_value_11")
                    max_value_12 = st.number_input("窑头煤粉max", value=6.968, format="%.4f", key="max_value_12")
                    max_value_13 = st.number_input("窑尾煤粉max", value=2.509, format="%.4f", key="max_value_13")


                    bounds_min = [min_value_1, min_value_2, min_value_3, min_value_4, min_value_5, min_value_6, min_value_7, min_value_8, min_value_9, min_value_10, min_value_11, min_value_12, min_value_13]
                    bounds_max = [max_value_1, max_value_2, max_value_3, max_value_4, max_value_5, max_value_6, max_value_7, max_value_8, max_value_9, max_value_10, max_value_11, max_value_12, max_value_13]
                    bounds = np.array(list(zip(bounds_min, bounds_max)))


                    st.sidebar.subheader("非操作变量边界设定")
                    col1, col2 = st.sidebar.columns(2)
                with col1:
                    min_value_a = st.number_input("磨机电流min", value=103.158, format="%.4f", key="min_value_a")
                    min_value_b = st.number_input("生料磨磨头入口温度温度min", value=383.922, format="%.4f", key="min_value_b")
                    min_value_c = st.number_input("生料磨磨尾入口温度min", value=398.151, format="%.4f", key="min_value_c")
                    min_value_d = st.number_input("磨压差min", value=162.28, format="%.4f", key="min_value_d")

                with col2:
                    max_value_a = st.number_input("磨机电流max", value=126.082, format="%.4f", key="max_value_a")
                    max_value_b = st.number_input("生料磨磨头入口温度温度max", value=469.238, format="%.4f", key="max_value_b")
                    max_value_c = st.number_input("生料磨磨尾入口温度max", value=486.629, format="%.4f", key="max_value_c")
                    max_value_d = st.number_input("磨压差max", value=198.33, format="%.4f", key="max_value_d")

                    minimum= [min_value_a, min_value_b, min_value_c, min_value_d]
                    maximum= [max_value_a, max_value_b, max_value_c, max_value_d]

                    st.sidebar.subheader("变量权重的确定")
                    col1, col2 = st.sidebar.columns(2)

                    with col1:
                        weight_options = {"极其重要": 1000000, "非常重要": 100000, "重要": 10000, "中等": 1000, "一般": 100, "不太重要": 10, "可忽略的": 1}
                        W_bounds_1 = st.selectbox("磨机电流权重", list(weight_options.keys()), index=0, key="W_bounds_1")
                        W_bounds_1 = weight_options[W_bounds_1]

                        W_bounds_2 = st.selectbox("生料磨磨头入口温度权重", list(weight_options.keys()), index=1, key="W_bounds_2")
                        W_bounds_2 = weight_options[W_bounds_2]

                    with col2:
                        W_bounds_3 = st.selectbox("生料磨磨尾入口温度权重", list(weight_options.keys()), index=2, key="W_bounds_3")
                        W_bounds_3 = weight_options[W_bounds_3]

                        W_bounds_4 = st.selectbox("磨压差权重", list(weight_options.keys()), index=3, key="W_bounds_4")
                        W_bounds_4 = weight_options[W_bounds_4]

                    W_bounds = [W_bounds_1, W_bounds_2, W_bounds_3, W_bounds_4]

                    W_min_boolean = np.array([1, 1, 1, 1])
                    W_max_boolean = np.array([1, 1, 1, 1])
                    
                    st.sidebar.subheader("变量单调性权重的确定")
                    col1, col2 = st.sidebar.columns(2)

                    with col1:
                        importance_options = {"重要": 10000, "中等": 1000, "一般": 100, "不太重要": 10, "可忽略的": 1}
                        W_monotony_1 = st.selectbox("磨机电流权重", list(importance_options.keys()), index=0, key="W_monotony_1")
                        W_monotony_1 = importance_options[W_monotony_1]

                        W_monotony_2 = st.selectbox("生料磨磨头入口温度权重", list(importance_options.keys()), index=1, key="W_monotony_2")
                        W_monotony_2 = importance_options[W_monotony_2]

                    with col2:
                        W_monotony_3 = st.selectbox("生料磨磨尾入口温度权重", list(importance_options.keys()), index=0, key="W_monotony_3")
                        W_monotony_3 = importance_options[W_monotony_3]

                        W_monotony_4 = st.selectbox("磨压差权重", list(importance_options.keys()), index=0, key="W_monotony_4")
                        W_monotony_4 = importance_options[W_monotony_4]

                    W_monotony = [W_monotony_1, W_monotony_2, W_monotony_3, W_monotony_4]
                    
                    st.sidebar.subheader("变量单调性确定")
                    col1, col2 = st.sidebar.columns(2)

                    with col1:
                        monotonicity_options = {"单调递增": -1, "单调递减": 1}
                        sign_1 = st.selectbox("磨机电流", list(monotonicity_options.keys()), index=0, key="sign_1")
                        sign_1 = monotonicity_options[sign_1]

                        sign_2 = st.selectbox("生料磨磨头入口温度", list(monotonicity_options.keys()), index=0, key="sign_2")
                        sign_2 = monotonicity_options[sign_2]

                    with col2:
                        sign_3 = st.selectbox("生料磨磨尾入口温度", list(monotonicity_options.keys()), index=0, key="sign_3")
                        sign_3 = monotonicity_options[sign_3]

                        sign_4 = st.selectbox("磨压差", list(monotonicity_options.keys()), index=1, key="sign_4")
                        sign_4 = monotonicity_options[sign_4]

                    sign = [sign_1, sign_2, sign_3, sign_4]
                    
                    W_target = np.array([0, 0, 0, 0])
                    target   = np.array([0, 0, 0, 0])
                    
                    st.sidebar.subheader("调整参数")
                    population_size = st.sidebar.number_input("种群大小", min_value=5, max_value=1000, value=20)
                    max_generations = st.sidebar.number_input("最大迭代次数", min_value=100, max_value=2000, value=200)
                    
                    models = None
                    model_indices = None
                    st.sidebar.subheader("上传模型文件")
                    model_files = st.sidebar.file_uploader("**磨机电流** > 生料磨磨头入口温度温度 > **生料磨磨尾入口温度** > 磨压差 ", accept_multiple_files=True, type=["pkl"])
                    if len(model_files) > 0:  # 检查文件列表长度是否大于0
                        models = [load_model(file) for file in model_files]
                        model_indices = [1, 2, 3, 4] 


                if st.button("Confirm"):
                    if model_files is not None:
                        crossover_rate = 0.8
                        mutation_factor = 0.5
                        # 初始化种群
                        population = initialize_population(population_size, bounds)
                        # 迭代优化
                        best_individual = None
                        best_fitness = float('inf')
                        with st.spinner('正在进行差分进化...'):
                            for generation in range(max_generations):
                                current_best_fitness = float('inf')
                                for individual in population:
                                    predictions = predict_data(np.array(individual.genes).reshape(1, -1), models, model_indices)
                                    predictions = predictions.flatten()
                                    fitness = objective_function(predictions, W_bounds, W_min_boolean, W_max_boolean, W_monotony, W_target, sign, minimum, maximum, mean, target, std)
                                    # 添加限制条件
                                    if np.isclose(np.sum(individual.genes[1:8]), 1):  # 检查genes[1:8]的和是否接近1
                                        individual.fitness = fitness
                                        if fitness < current_best_fitness:
                                            current_best_fitness = fitness
                                            current_best_individual = individual
                                    else:
                                        # 如果genes[1:8]的和不为1，则调整为1
                                        individual.genes[1:8] /= np.sum(individual.genes[1:8])
                                        predictions = predict_data(np.array(individual.genes).reshape(1, -1), models, model_indices)
                                        predictions = predictions.flatten()
                                        fitness = objective_function(predictions, W_bounds, W_min_boolean, W_max_boolean, W_monotony, W_target, sign, minimum, maximum, mean, target, std)
                                        individual.fitness = fitness
                                        if fitness < current_best_fitness:
                                            current_best_fitness = fitness
                                            current_best_individual = individual
                                    print("Genes:", individual.genes)
                                    print("Predictions:", predictions)
                                    print("Fitness:", fitness)
                                # 更新全局最佳个体和最佳目标函数值
                                if current_best_fitness < best_fitness:
                                    best_fitness = current_best_fitness
                                    best_individual = current_best_individual

                                # 进化操作
                                population = evolve(population, objective_function, crossover_rate, mutation_factor, bounds)
                            # 输出最终的最佳个体和最佳目标函数值
                            st.subheader("Best Individual:")
                            if best_individual is not None:
                                st.write("Recommended Operation Variables:")
                                for i, variable_name in enumerate(column_names_fixed):
                                    st.write(f"{variable_name}: {best_individual.genes[i]:.4f}")
                                st.write("Minimum Fitness of the Model: {:.4f}".format(best_fitness))
                                st.success("Best individual and minimum fitness found successfully!")
                            else:
                                st.write("No best individual found.")
                                st.error("Failed to find the best individual.")
            col1, col2 = st.columns(2)

            if col1.button("返回第一步"):
                st.session_state.page_num = 1
                st.rerun()

            if col2.button("返回"):
                st.session_state.page_num = 3
                st.rerun()     

        if __name__ == "__main__":
            main()

# 主程序
def main():
    if st.session_state.get('page_num', 1) == 1:
        page1()
    elif st.session_state.get('page_num', 1) == 2:
        page2()
    elif st.session_state.get('page_num', 1) == 3:
        page3()
    elif st.session_state.get('page_num', 1) == 4:
        page4()
    elif st.session_state.get('page_num', 1) == 5:
        page5()
    elif st.session_state.get('page_num', 1) == 6:
        page6()  
    elif st.session_state.get('page_num', 1) == 7:
        page7()
if __name__ == '__main__':
    main()
