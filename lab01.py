#1. Definicao das bibliotecas
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn import model_selection
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from warnings import filterwarnings
import random
from lab03 import enviar_email
from tqdm import tqdm
import cores as COR
import seaborn as sns
from sklearn.metrics import accuracy_score
from lab02 import escreva_relatorio
filterwarnings('ignore')

#2. Definicao da semente para geracao de numereos aleatorios
#Intialise a random number generator
# Set a seed value
seed_value= 12321
# 1. Set 'PYTHONHASHSEED' environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
# 2. Set 'python' built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

#3. Leitura dos dados
dataframe = pd.read_csv("Dry_Bean_Dataset.csv")

# Pegando os nomes das colunas, exceto a última
column_names = dataframe.columns[:-1]

# Convertendo para um array
column_names_array = column_names.to_numpy()

#4. A analise exploratoria dos dados realizada em outro script
# Visão geral dos dados

# Converter colunas para float e tratar erros
columns_to_convert = ['Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'AspectRation', 'Eccentricity',
					  'EquivDiameter', 'Extent', 'Solidity', 'roundness', 'Compactness', 'ShapeFactor1',
					  'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4']

#for column in columns_to_convert:
for column in column_names_array:
	dataframe[column] = pd.to_numeric(dataframe[column], errors='coerce')

#5. Preparacao dos dados
print(COR.AMA, "Apresentando o shape dos dados (dimenssoes)")
print(dataframe.shape)


# Tratar valores NaN resultantes da conversão
# Você pode optar por preencher com a média, mediana, zero ou excluir as linhas/colunas
# Exemplo: preenchendo com zero
dataframe.fillna(0, inplace=True)

array = dataframe.values
X = array[:,0:16]
Y = array[:,16]

#6. Divisao da base de dados em treinamento, validacao e teste
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=seed_value)

#X_train_p, X_valid, y_train_p, y_valid = train_test_split(X_train, y_train, random_state=seed)


#7. Realizar busca com o gridsearch ou randonsearch para encontrar os melhores parametros de cada modelo
def rodar_modelos(rodada):
	detalhes_info = ''

	# define models
	decisionTree = DecisionTreeClassifier()
	svc = SVC()
	
	# define evaluation
	cv = model_selection.StratifiedKFold(n_splits=10)
	
	# define search space for decision tree and range for svm
	space, param_grid = parametros_iniciais()

	params_choice = f"========================================================\n"
	params_choice += f"                     Parâmetros Iniciais - Rodada {rodada}\n"
	params_choice += f"========================================================\n\n"
	params_choice += f"Parametros para árvore de decisão:\n"
	params_choice += f"criterion: {space['criterion']}\n"
	params_choice += f"min_samples_split: {space['min_samples_split']}\n"
	params_choice += f"max_depth: {space['max_depth']}\n"
	params_choice += f"min_samples_leaf: {space['min_samples_leaf']}\n"
	params_choice += f"Parametros para SVM:\n"
	params_choice += f"param_grid C: {param_grid['C']}\n"
	params_choice += f"param_grid gamma: {param_grid['gamma']}\n"
	params_choice += f"param_grid kernel: {param_grid['kernel']}\n"
	params_choice += f"========================================================\n\n"
	print(COR.AZU, params_choice)
	
	# define random search for decision tree
	search = RandomizedSearchCV(decisionTree, space, n_iter=50, scoring='accuracy', n_jobs=4, cv=cv)
	
	# execute search
	#result_tree = search.fit(X_train, y_train)

	# Loop para simular a barra de progresso
	for _ in tqdm(range(50), desc="Progresso da Árvore de Decisão"):
		result_tree = search.fit(X_train, y_train)
	
	# summarize result for decision tree
	print(COR.VERD, '=========Random Search Results for TREE==========')
	print('Best Score: %s' % result_tree.best_score_)
	print('Best Hyperparameters: %s' % result_tree.best_params_)
	detalhes_info += '\n=========Random Search Results for TREE=========='
	detalhes_info += '\nBest Score: %s' % result_tree.best_score_
	detalhes_info += '\nBest Hyperparameters: %s' % result_tree.best_params_

	# define random search for SVM
	search = RandomizedSearchCV(svc, param_grid, n_iter=10, scoring='accuracy', n_jobs=4, cv=cv, random_state=seed_value)
	
	# execute search
	#result_svc = search.fit(X_train, y_train)

	# Loop para simular a barra de progresso
	for _ in tqdm(range(10), desc="Progresso do SVM"):
		result_svc = search.fit(X_train, y_train)
	
	# summarize result for SVM
	print(COR.VERM, '\n=========Random Search Results for SVM==========')
	print('Best Score: %s' % result_svc.best_score_)
	print('Best Hyperparameters: %s' % result_svc.best_params_)
	detalhes_info += '\n\n=========Random Search Results for SVM=========='
	detalhes_info += '\nBest Score: %s' % result_svc.best_score_
	detalhes_info += '\nBest Hyperparameters: %s' % result_svc.best_params_
	detalhes_info += '\n\n========================================================\n'

	#8. Definicao dos modelos de classificacao com as melhores configuracoes
	# criacao dos modelos com os melhores parametros
	RFC = RandomForestClassifier(n_estimators=30, random_state=seed_value)
	svc = result_svc.best_estimator_
	DTC = result_tree.best_estimator_   #tree.DecisionTreeClassifier(criterion='entropy', random_state=seed)
	MLP = MLPClassifier(hidden_layer_sizes=(100, 100), random_state=seed_value)
	BMLP = BaggingClassifier(base_estimator=MLPClassifier(hidden_layer_sizes=(5, 15, 5), random_state=seed_value),
							 n_estimators=20, random_state=seed_value)
	
	#adiciona os modelos em uma lista
	models = []
	models.append(('Arvore', DTC))
	models.append(('SVM', svc))
	models.append(('ComiteArvore', RFC))
	models.append(('RedeNeural', MLP))
	models.append(('ComiteRede', BMLP))
	
	# evaluate each model in turn
	results = []
	names = []
	
	#deficao da metrica a ser utilizada
	scoring = 'accuracy'
	
	#9. Definicao do modelo experimental
	#amostragem estratificada
	#kfold = cv
	
	#10 Execucao do modelo experimental
	#avaliacao de cada modelo nas amotragens estratificas
	print(COR.AMA, '\nDesempenhos medios dos modelos:')
	detalhes_info += '\n\nDesempenhos medios dos modelos:'
	for name, model in models:
		cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=10, scoring=scoring)
		results.append(cv_results)
		names.append(name)
		msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
		print(msg)
		detalhes_info += f'\n{msg}'
	
	#11 Comparacao de modelos
	# Teste de hipotese analisando o p-value
	stat, p = stats.kruskal(results[0],results[1],results[2],results[3],results[4])
	alpha = 0.05
	if p > alpha:
		print(COR.VERD, '\nSame distributions (fail to reject H0)')
		detalhes_info += '\n\nSame distributions (fail to reject H0)'

	else:
		print(COR.VERM, '\nDifferent distributions (reject H0)')
		detalhes_info += '\n\nDifferent distributions (reject H0)'

	
	print(COR.AZU, '\nComparison stats', stat)
	
	print('Comparacao Arvore | SVM ->', stats.kruskal(results[0], results[1]))
	print('Comparacao Arvore | ComiteArvore ->', stats.kruskal(results[0], results[2]))
	print('Comparacao Arvore | RedeNeural ->', stats.kruskal(results[0],results[3]))
	print('Comparacao Arvore | CRNA ->', stats.kruskal(results[0], results[4]))
	print('Comparacao SVM | RedeNeural ->', stats.kruskal(results[1], results[3]))
	print('Comparacao SVM | ComiteArvore ->', stats.kruskal(results[1], results[2]))
	print('Comparacao SVM | ComiteRede ->', stats.kruskal(results[1], results[4]))
	print('Comparacao ComiteArvore | ComiteRede ->', stats.kruskal(results[2], results[4]))
	print('Comparacao ComiteArvore | RedeNeural ->', stats.kruskal(results[2], results[3]))
	print('Comparacao RedeNeural | ComiteRede ->', stats.kruskal(results[3], results[4]))

	detalhes_info += f'\n\nComparison stats {stat}'

	detalhes_info += f'\nComparacao Arvore | SVM -> {stats.kruskal(results[0], results[1])}'
	detalhes_info += f'\nComparacao Arvore | ComiteArvore -> {stats.kruskal(results[0], results[2])}'
	detalhes_info += f'\nComparacao Arvore | RedeNeural -> {stats.kruskal(results[0], results[3])}'
	detalhes_info += f'\nComparacao Arvore | CRNA -> {stats.kruskal(results[0], results[4])}'
	detalhes_info += f'\nComparacao SVM | RedeNeural -> {stats.kruskal(results[1], results[3])}'
	detalhes_info += f'\nComparacao SVM | ComiteArvore -> {stats.kruskal(results[1], results[2])}'
	detalhes_info += f'\nComparacao SVM | ComiteRede -> {stats.kruskal(results[1], results[4])}'
	detalhes_info += f'\nComparacao ComiteArvore | ComiteRede -> {stats.kruskal(results[2], results[4])}'
	detalhes_info += f'\nComparacao ComiteArvore | RedeNeural -> { stats.kruskal(results[2], results[3])}'
	detalhes_info += f'\nComparacao RedeNeural | ComiteRede -> {stats.kruskal(results[3], results[4])}'

	#treinamento dos modelos no conjunto de treino completo (sem divisao de validacao)
	RFC.fit(X_train, y_train)
	svc.fit(X_train, y_train)
	DTC.fit(X_train, y_train)
	MLP.fit(X_train, y_train)
	BMLP.fit(X_train, y_train)
	
	#predicao de cada modelo para a base de teste
	Y_test_prediction_RFC = RFC.predict(X_test)
	Y_test_prediction_SVC = svc.predict(X_test)
	Y_test_prediction_DTC = DTC.predict(X_test)
	Y_test_prediction_MLP = MLP.predict(X_test)
	Y_test_prediction_BMLP = BMLP.predict(X_test)
	
	#12 Apresentacao de resultados
	print(COR.AMA, "\nAcuracia Comite de Arvore: Treinamento",  RFC.score(X_train, y_train)," Teste" ,RFC.score(X_test, y_test))
	print("Clasification report:", classification_report(y_test, Y_test_prediction_RFC))
	print("Confussion matrix:\n", confusion_matrix(y_test, Y_test_prediction_RFC))
	
	print("\nAcuracia SVC: Treinamento",  svc.score(X_train, y_train)," Teste" ,svc.score(X_test, y_test))
	print("Clasification report:", classification_report(y_test, Y_test_prediction_SVC))
	print("Confussion matrix:\n", confusion_matrix(y_test, Y_test_prediction_SVC))
	
	print("\nAcuracia Arvore: Treinamento",  DTC.score(X_train, y_train)," Teste" ,DTC.score(X_test, y_test))
	print("Clasification report:", classification_report(y_test, Y_test_prediction_DTC))
	print("Confussion matrix:\n", confusion_matrix(y_test, Y_test_prediction_DTC))
	
	print("\nAcuracia Rede Neural: Treinamento",  MLP.score(X_train, y_train)," Teste" ,MLP.score(X_test, y_test))
	print("Clasification report:", classification_report(y_test, Y_test_prediction_MLP))
	print("Confussion matrix:\n", confusion_matrix(y_test, Y_test_prediction_MLP))
	
	print("\nAcuracia Comite RNA: Treinamento",  BMLP.score(X_train, y_train)," Teste" ,BMLP.score(X_test, y_test))
	print("Clasification report:", classification_report(y_test, Y_test_prediction_BMLP))
	print("Confussion matrix:\n", confusion_matrix(y_test, Y_test_prediction_BMLP), COR.FIM)

	detalhes_info += f'\n\nAcuracia Comite de Arvore: Treinamento {RFC.score(X_train, y_train)}, Teste {RFC.score(X_test, y_test)}'
	detalhes_info += f'\nClasification report: {classification_report(y_test, Y_test_prediction_RFC)}'
	detalhes_info += f'\nConfussion matrix:\n {confusion_matrix(y_test, Y_test_prediction_RFC)}'

	detalhes_info += f'\n\nAcuracia SVC: Treinamento", {svc.score(X_train, y_train)}, Teste {svc.score(X_test, y_test)}'
	detalhes_info += f'\nClasification report: {classification_report(y_test, Y_test_prediction_SVC)}'
	detalhes_info += f'\nConfussion matrix:\n {confusion_matrix(y_test, Y_test_prediction_SVC)}'

	detalhes_info += f'\n\nAcuracia Arvore: Treinamento", {DTC.score(X_train, y_train)}, Teste {DTC.score(X_test, y_test)}'
	detalhes_info += f'\nClasification report:", {classification_report(y_test, Y_test_prediction_DTC)}'
	detalhes_info += f'\nConfussion matrix:\n {confusion_matrix(y_test, Y_test_prediction_DTC)}'

	detalhes_info += f'\n\nAcuracia Rede Neural: Treinamento", {MLP.score(X_train, y_train)}, Teste {MLP.score(X_test, y_test)}'
	detalhes_info += f'\nClasification report: {classification_report(y_test, Y_test_prediction_MLP)}'
	detalhes_info += f'\nConfussion matrix:\n {confusion_matrix(y_test, Y_test_prediction_MLP)}'

	detalhes_info += f'\n\nAcuracia Comite RNA: Treinamento {BMLP.score(X_train, y_train)}, Teste {BMLP.score(X_test, y_test)}'
	detalhes_info += f'\nClasification report: {classification_report(y_test, Y_test_prediction_BMLP)}'
	detalhes_info += f'\nConfussion matrix:\n {confusion_matrix(y_test, Y_test_prediction_BMLP)}'

	fig = plt.figure()
	fig.suptitle('Algorithm Comparison')
	ax = fig.add_subplot(111)
	plt.boxplot(results)
	ax.set_xticklabels(names)

	# Obtém a data e hora atual
	current_time = datetime.datetime.now()
	# Formata a data e hora no formato desejado (por exemplo, YYYY-MM-DD_HH-MM-SS)
	formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
	# Define o nome do arquivo com a data e hora
	filename = f"relatorios/Algorithm_Comparison_{formatted_time}.png"

	escreva_relatorio(rodada, space, param_grid, detalhes_info, formatted_time)

	# Salva a figura
	plt.savefig(filename)

	#plt.show()

	mensagem = params_choice
	enviar_email(f"Finalizada rodada {rodada}", mensagem)
	

def parametros_iniciais():
	space = dict()

	# Definindo listas de possíveis valores para cada parâmetro
	possible_splits = list(range(2, 21)) # Pode ir de 2 até o número de amostras
	possible_depths = list(range(1, 21)) # Pode ir de 1 até um número prático
	possible_leafs = list(range(1, 21)) # Pode ir de 1 até um número prático

	# Escolhendo valores aleatórios sem repetição
	space['criterion'] = ['gini', 'entropy']
	space['min_samples_split'] = random.sample(possible_splits, 4)
	space['max_depth'] = random.sample(possible_depths, 10)
	space['min_samples_leaf'] = random.sample(possible_leafs, 2)

	# Definindo o intervalo de parâmetros para svm
	possible_C = [0.01, 0.1, 1, 10, 100] # Exemplo de intervalo mais amplo
	possible_gamma = [1, 0.1, 0.01, 0.001, 0.0001] # Exemplo de intervalo mais amplo
	possible_kernels = ['rbf'] # Todos os kernels comuns

	param_grid = {
		'C': random.sample(possible_C, 3),
		'gamma': random.sample(possible_gamma, 4),
		'kernel': [random.choice(possible_kernels)]
	}

	return space, param_grid


rodadas = 10
try:
	for i in range(rodadas):
		rodar_modelos(i + 1)
except Exception as e:
	# Convertendo o erro para string antes de enviar
	mensagem_erro = str(e)
	enviar_email("Ocorreu um erro no script", mensagem_erro)


