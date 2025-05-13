import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import ttest_ind
import os
import kagglehub
from scipy.stats import chi2_contingency
import plotly.express as px


# Kaggle
path = kagglehub.dataset_download("pavansubhasht/ibm-hr-analytics-attrition-dataset")

# Listar arquivos no diretório especificado
arquivos_no_diretorio = os.listdir(path)
print("Arquivos no diretório:", arquivos_no_diretorio)


caminho_arquivo_csv = os.path.join(path, 'WA_Fn-UseC_-HR-Employee-Attrition.csv')

# Verificar se o arquivo CSV existe
if os.path.exists(caminho_arquivo_csv):
    # Ler o arquivo CSV
    df = pd.read_csv(caminho_arquivo_csv)
    print("Arquivo CSV carregado com sucesso no DataFrame.")

    # Mostrar o DataFrame
    print("\nPré-visualização dos dados:")

else:
    print(f"Erro: Arquivo CSV não encontrado no caminho esperado: {caminho_arquivo_csv}")
    print("Por favor, verifique a lista de arquivos acima para encontrar o nome correto.")

df.to_csv('attrition_data.csv', index=False)

traducao_colunas = {
    'Age': 'Idade',
    'Attrition': 'Rotatividade (Se deixou a empresa)',
    'BusinessTravel': 'Viagens a trabalho',
    'DailyRate': 'Taxa diária (remuneração)',
    'Department': 'Departamento',
    'DistanceFromHome': 'Distância de casa (km)',
    'Education': 'Nível de educação',
    'EducationField': 'Área de formação',
    'EmployeeCount': 'Contagem de funcionários',
    'EmployeeNumber': 'Número do funcionário',
    'EnvironmentSatisfaction': 'Satisfação com o ambiente',
    'Gender': 'Gênero',
    'HourlyRate': 'Taxa horária (remuneração)',
    'JobInvolvement': 'Envolvimento no trabalho',
    'JobLevel': 'Nível do cargo',
    'JobRole': 'Cargo',
    'JobSatisfaction': 'Satisfação no trabalho',
    'MaritalStatus': 'Estado civil',
    'MonthlyIncome': 'Renda mensal',
    'MonthlyRate': 'Taxa mensal (remuneração)',
    'NumCompaniesWorked': 'Número de empresas onde trabalhou',
    'Over18': 'Maior de 18 anos',
    'OverTime': 'Horas extras',
    'PercentSalaryHike': 'Percentual de aumento salarial',
    'PerformanceRating': 'Avaliação de desempenho',
    'RelationshipSatisfaction': 'Satisfação com relacionamentos',
    'StandardHours': 'Horas padrão de trabalho',
    'StockOptionLevel': 'Nível de opções de ações',
    'TotalWorkingYears': 'Total de anos trabalhados',
    'TrainingTimesLastYear': 'Treinamentos no último ano',
    'WorkLifeBalance': 'Equilíbrio vida-trabalho',
    'YearsAtCompany': 'Anos na empresa',
    'YearsInCurrentRole': 'Anos no cargo atual',
    'YearsSinceLastPromotion': 'Anos desde última promoção',
    'YearsWithCurrManager': 'Anos com o gestor atual'
}

df = df.rename(columns=traducao_colunas)

# Configuração de visualização
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Converter Rotatividade para binário
df['Rotatividade_num'] = df['Rotatividade (Se deixou a empresa)'].apply(lambda x: 1 if x == 'Yes' else 0)

# Correlação com variáveis numéricas
correlacoes = df.corr(numeric_only=True)['Rotatividade_num'].sort_values(ascending=False)
print("Variáveis mais correlacionadas com Rotatividade:\n", correlacoes)

# Gráfico
plt.figure(figsize=(10, 6))
sns.heatmap(df[['Rotatividade_num', 'Satisfação no trabalho', 'Renda mensal', 'Idade']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlação entre Rotatividade e Variáveis-Chave')
plt.show()

#rotatividade x horas extras
tab = pd.crosstab(df['Horas extras'], df['Rotatividade (Se deixou a empresa)'], normalize='index') * 100
print("Taxa de Rotatividade por Horas Extras:\n", tab)

# Teste Chi-quadrado
chi2, p, _, _ = chi2_contingency(pd.crosstab(df['Horas extras'], df['Rotatividade (Se deixou a empresa)']))
print(f"Teste Chi-quadrado: p-value = {p:.4f}")

# Gráfico
sns.barplot(x='Horas extras', y='Rotatividade_num', data=df)
plt.title('Taxa de Rotatividade por Horas Extras')
plt.ylabel('Probabilidade de Rotatividade (%)')
plt.show()

#Rotatividade por departamento
rotatividade_depto = df.groupby('Departamento')['Rotatividade_num'].mean().sort_values(ascending=False)
print("Taxa de Rotatividade por Departamento:\n", rotatividade_depto)

# Gráfico
rotatividade_depto.plot(kind='bar', color='salmon')
plt.title('Taxa de Rotatividade por Departamento')
plt.ylabel('Taxa (%)')
plt.xticks(rotation=45)
plt.show()

# "Promovido nos últimos 3 anos"
df['Promovido_recente'] = df['Anos desde última promoção'].apply(lambda x: 'Sim' if x <= 3 else 'Não')

# Comparar rotatividade
sns.countplot(x='Promovido_recente', hue='Rotatividade (Se deixou a empresa)', data=df)
plt.title('Rotatividade: Promovidos vs Não Promovidos (últimos 3 anos)')
plt.show()

# satisfacao x gesto
sns.lmplot(x='Anos com o gestor atual', y='Satisfação no trabalho', data=df,
           line_kws={'color': 'red'}, scatter_kws={'alpha': 0.3})
plt.title('Relação entre Tempo com o Gestor e Satisfação')
plt.show()

#viagens de trabalho x rotatividade
sns.catplot(x='Viagens a trabalho', y='Rotatividade_num', kind='bar', data=df)
plt.title('Taxa de Rotatividade por Frequência de Viagens')
plt.show()

# salario x expe
sns.scatterplot(x='Total de anos trabalhados', y='Renda mensal', hue='Cargo', data=df, alpha=0.6)
plt.title('Renda Mensal vs Experiência por Cargo')
plt.show()

#treinamento e desempenho
sns.lineplot(x='Treinamentos no último ano', y='Avaliação de desempenho', data=df, marker='o')
plt.title('Desempenho Médio por Número de Treinamentos')
plt.show()

#estado civil x rotatividade
pd.crosstab(df['Estado civil'], df['Rotatividade (Se deixou a empresa)'], normalize='index').plot(kind='bar', stacked=True)
plt.title('Proporção de Rotatividade por Estado Civil')
plt.ylabel('Proporção')
plt.show()

# idade x satisfacao
sns.regplot(x='Idade', y='Satisfação no trabalho', data=df, lowess=True, line_kws={'color': 'red'})
plt.title('Relação entre Idade e Satisfação no Trabalho')
plt.show()

# diferenca salarial por genero
# Configurações do gráfico
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid", palette="pastel")

# dados
top_cargos = df['Cargo'].unique()[:5]
data_to_plot = []

for cargo in top_cargos:
    subset = df[df['Cargo'] == cargo]
    male_sal = subset[subset['Gênero'] == 'Male']['Renda mensal']
    female_sal = subset[subset['Gênero'] == 'Female']['Renda mensal']


    ci_male = 1.96 * (male_sal.std() / np.sqrt(len(male_sal))) if len(male_sal) > 0 else 0
    ci_female = 1.96 * (female_sal.std() / np.sqrt(len(female_sal))) if len(female_sal) > 0 else 0

    t_stat, p_val = ttest_ind(male_sal, female_sal) if len(male_sal) > 1 and len(female_sal) > 1 else (np.nan, np.nan)


    data_to_plot.append({
        'Cargo': cargo,
        'Masculino': male_sal.mean() if len(male_sal) > 0 else 0,
        'Feminino': female_sal.mean() if len(female_sal) > 0 else 0,
        'CI_M': ci_male,
        'CI_F': ci_female,
        'p_value': p_val
    })

# Criar DataFrame para plotagem
plot_df = pd.DataFrame(data_to_plot)

# Ordenar pela maior diferença
plot_df['Diferença'] = plot_df['Masculino'] - plot_df['Feminino']
plot_df = plot_df.sort_values('Diferença', ascending=False)

# Gráfico de barras
bar_width = 0.35
index = np.arange(len(plot_df))

fig, ax = plt.subplots()
bars1 = ax.bar(index, plot_df['Masculino'], bar_width,
               yerr=plot_df['CI_M'], label='Masculino', color='skyblue')
bars2 = ax.bar(index + bar_width, plot_df['Feminino'], bar_width,
               yerr=plot_df['CI_F'], label='Feminino', color='salmon')

# Adicionar rótulos, título, legendas, etc. para completar o gráfico
ax.set_xlabel('Cargo')
ax.set_ylabel('Renda Mensal Média')
ax.set_title('Diferença Salarial Média por Gênero para Top 5 Cargos')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(plot_df['Cargo'])
ax.legend()
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# distancia de casa x rotatividade
df['Faixa_distancia'] = pd.cut(df['Distância de casa (km)'], bins=[0, 10, 20, 30, 100], right=False, labels=['0-10 km', '10-20 km', '20-30 km', '30+ km'])
sns.barplot(x='Faixa_distancia', y='Rotatividade_num', data=df)
plt.title('Taxa de Rotatividade por Distância de Casa')
plt.show()


# engajamento x tempo de empresa
sns.heatmap(pd.pivot_table(df, index='Anos na empresa', columns='Envolvimento no trabalho',
                           values='Rotatividade_num', aggfunc='mean'),
            cmap='YlOrRd', annot=True)
plt.title('Taxa Média de Rotatividade por Anos na Empresa e Nível de Engajamento')
plt.show()

# salvar as ideias e exportar
plt.savefig('insight1.png', dpi=300, bbox_inches='tight')

# Salva o DataFrame rotatividade_depto em um arquivo CSV
rotatividade_depto.to_csv('Projeto_RH.csv')
df.to_csv('Projeto_RH.csv', index=False, mode='a', header=False, encoding='utf-8-sig')

# grafico interativo
fig = px.scatter(df, x='Idade', y='Renda mensal', color='Rotatividade (Se deixou a empresa)', hover_data=['Cargo'])
fig.show()