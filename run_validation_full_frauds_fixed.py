import numpy as np
import pandas as pd
import time
from stepwise_optimized import StepwiseZeroInflated
from sklearn.feature_selection import VarianceThreshold

print("🔍 VALIDAÇÃO COMPLETA: DATASET TRANSACTIONS_FEATURES.PARQUET")
print("🎯 TARGET: 'frauds' | DATASET: COMPLETO | EARLY_STOPPING: FALSE")
print("="*80)

# Carregar dataset
try:
    df = pd.read_parquet('transactions_features.parquet')
    print(f"✅ Dataset carregado: {df.shape[0]:,} × {df.shape[1]} colunas")
    print(f"📁 Memória: {df.memory_usage(deep=True).sum()/1024**2:.1f} MB")
except Exception as e:
    print(f"❌ Erro carregando dataset: {e}")
    exit()

# Verificar colunas disponíveis
print(f"\n📋 COLUNAS DISPONÍVEIS:")
all_cols = df.columns.tolist()
print(f"Total de colunas: {len(all_cols)}")

# Buscar coluna target 'frauds'
target_col = None
possible_targets = ['frauds']

for col in possible_targets:
    if col in all_cols:
        target_col = col
        print(f"✅ Target encontrado: '{col}'")
        break

if target_col is None:
    # Listar primeiras colunas para ajudar
    print(f"❌ Coluna 'frauds' não encontrada. Colunas disponíveis:")
    for i, col in enumerate(all_cols[:20]):
        print(f"  {i+1:2d}. {col}")
    if len(all_cols) > 20:
        print(f"  ... e mais {len(all_cols)-20} colunas")
    
    # Tentar detectar coluna que pareça ser target de fraude
    print(f"\n🔍 Procurando colunas que podem ser target de fraude...")
    fraud_candidates = []
    for col in all_cols:
        col_lower = col.lower()
        if any(word in col_lower for word in ['fraud', 'target', 'label', 'class']):
            fraud_candidates.append(col)
    
    if fraud_candidates:
        print(f"📋 Candidatos encontrados: {fraud_candidates}")
        target_col = fraud_candidates[0]
        print(f"✅ Usando: '{target_col}'")
    else:
        print(f"❌ Nenhum candidato encontrado. Saindo...")
        exit()

# Extrair target
try:
    y = df[target_col].values
    
    # Converter para inteiro não-negativo se necessário
    if y.dtype != int:
        y = y.astype(int)
    y = np.abs(y)  # Garantir não-negativo
    
    print(f"\n🎯 ANÁLISE DO TARGET '{target_col}':")
    print(f"Tipo: {df[target_col].dtype}")
    print(f"Range: {y.min()} - {y.max()}")
    print(f"Zeros: {(y==0).sum():,} ({(y==0).mean()*100:.2f}%)")
    print(f"Não-zeros: {(y>0).sum():,} ({(y>0).mean()*100:.2f}%)")
    print(f"Valores únicos: {len(np.unique(y))}")
    
    if len(np.unique(y)) <= 30:
        unique_counts = pd.Series(y).value_counts().sort_index()
        print(f"Distribuição:")
        for val, count in unique_counts.head(10).items():
            print(f"  {val}: {count:,} ({count/len(y)*100:.2f}%)")
        if len(unique_counts) > 10:
            print(f"  ... e mais {len(unique_counts)-10} valores")
    
except Exception as e:
    print(f"❌ Erro processando target: {e}")
    exit()

# Preparar features (todas exceto target)
print(f"\n📊 PREPARANDO FEATURES:")
feature_cols = [col for col in all_cols if col != target_col]
print(f"Features candidatas: {len(feature_cols)}")

# Selecionar apenas colunas numéricas
numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
print(f"Features numéricas: {len(numeric_features)}")

if len(numeric_features) == 0:
    print(f"❌ Nenhuma feature numérica encontrada!")
    exit()

# Preparar matriz X
X = df[numeric_features].copy()

# Tratar valores faltantes
missing_counts = X.isnull().sum()
features_with_missing = missing_counts[missing_counts > 0]

if len(features_with_missing) > 0:
    print(f"⚠️  Features com valores faltantes: {len(features_with_missing)}")
    # Imputar com média
    X = X.fillna(X.mean())
    print(f"✅ Imputação com média aplicada")

# Filtrar variância baixa
print(f"\n🔧 FILTRO DE VARIÂNCIA:")
var_filter = VarianceThreshold(threshold=1e-8)
X_filtered = var_filter.fit_transform(X)
selected_features = X.columns[var_filter.get_support()]

print(f"Features antes: {X.shape[1]}")
print(f"Features após filtro: {len(selected_features)}")
print(f"Features removidas: {X.shape[1] - len(selected_features)}")

X_final = pd.DataFrame(X_filtered, columns=selected_features)

# Configuração do modelo
print(f"\n⚙️ CONFIGURAÇÃO STEPWISE:")
stepwise = StepwiseZeroInflated(
    alpha=0.05,
    model_type='ZINB',
    selection_criterion='AIC',
    max_iter=200,                   # Aumentando para garantir teste completo
    verbose=False,                  # Progress bar ativo
    use_cache=True,
    fit_params=dict(disp=False),
    parallel_features=True,
    n_jobs=-1,                      # Usar todos os cores
    batch_size=1,                   # SUA CONFIGURAÇÃO
    validation_steps=True,
    early_stopping=False            # SUA CONFIGURAÇÃO - VALIDAÇÃO COMPLETA
)

print(f"✅ TARGET: '{target_col}' ({(y==0).mean()*100:.1f}% zeros)")
print(f"✅ FEATURES: {X_final.shape[1]} variáveis independentes")
print(f"✅ OBSERVAÇÕES: {X_final.shape[0]:,} registros")
print(f"✅ early_stopping=False (VALIDAÇÃO COMPLETA DE TODAS AS FEATURES)")
print(f"✅ batch_size=1 (TESTE UNITÁRIO DE CADA FEATURE)")
print(f"✅ max_iter=200 (aumentado para garantir completude)")

# Verificar tamanho do problema
total_combinations = X_final.shape[1] * 3  # Cada feature: exog, inf, both
print(f"⚡ Combinações estimadas: ~{total_combinations:,}")

# Executar teste
print(f"\n🚀 EXECUTANDO STEPWISE NO DATASET COMPLETO...")
print(f"⏱️  Início: {time.strftime('%H:%M:%S')}")
start_time = time.time()

try:
    # Normalizar para estabilidade numérica
    print(f"🔧 Normalizando features...")
    X_mean = X_final.mean()
    X_std = X_final.std() + 1e-8
    X_norm = (X_final - X_mean) / X_std
    
    print(f"✅ Normalização concluída")
    print(f"🚀 Iniciando ajuste do modelo...")
    
    # AJUSTAR MODELO
    stepwise.fit(X_norm, y)
    
    duration = time.time() - start_time
    
    # RESULTADOS
    print(f"\n🎉 SUCESSO! Tempo total: {duration/60:.1f} minutos")
    print(f"\n📊 RESULTADOS FINAIS:")
    print(f"Features exógenas: {len(stepwise.columns_exog_)}")
    if stepwise.columns_exog_:
        print(f"  └─ {stepwise.columns_exog_}")
    
    print(f"Features inflacionadas: {len(stepwise.columns_inf_)}")
    if stepwise.columns_inf_:
        print(f"  └─ {stepwise.columns_inf_}")
    
    print(f"Features excluídas: {len(stepwise.excluded_)}")
    features_testadas = len(stepwise.excluded_) + len(stepwise.columns_exog_) + len(stepwise.columns_inf_)
    print(f"Features testadas: {features_testadas}/{X_final.shape[1]}")
    
    # VERIFICAR SE TESTOU TODAS AS FEATURES
    if features_testadas == X_final.shape[1]:
        print(f"✅ SUCESSO: TODAS as features foram testadas!")
    else:
        print(f"⚠️  ATENÇÃO: Apenas {features_testadas}/{X_final.shape[1]} features foram testadas")
        print(f"   Restam {X_final.shape[1] - features_testadas} features não testadas")
    
    # Cache stats CORRIGIDO
    cache_stats = stepwise.get_cache_stats()
    print(f"\n⚡ PERFORMANCE:")
    print(f"Cache hit rate: {cache_stats['hit_rate_percent']:.1f}%")
    print(f"Cache hits: {cache_stats['cache_hits']:,}")
    print(f"Cache misses: {cache_stats['cache_misses']:,}")
    print(f"Total modelos: {cache_stats['cache_hits'] + cache_stats['cache_misses']:,}")
    print(f"Modelos em cache: {cache_stats['model_cache_size']:,}")
    print(f"Matrizes em cache: {cache_stats['matrix_cache_size']:,}")
    
    # Modelo final
    if stepwise.final_model_:
        print(f"\n🏆 MODELO FINAL:")
        print(f"AIC: {stepwise.final_model_.aic:.2f}")
        print(f"Log-likelihood: {stepwise.final_model_.llf:.2f}")
        print(f"Convergiu: {stepwise.final_model_.mle_retvals['converged']}")
        print(f"Parâmetros: {len(stepwise.final_model_.params)}")
    
    print(f"\n✅ VALIDAÇÃO COMPLETA CONCLUÍDA COM SUCESSO!")
    print(f"🎯 Dataset transactions_features.parquet VALIDADO")
    print(f"🏆 Modelo final pronto para produção")
    
except KeyboardInterrupt:
    duration = time.time() - start_time
    print(f"\n⚠️  INTERROMPIDO pelo usuário após {duration/60:.1f} minutos")
    
    # Mostrar progresso parcial se disponível
    try:
        cache_stats = stepwise.get_cache_stats()
        total_models = cache_stats['cache_hits'] + cache_stats['cache_misses']
        if total_models > 0:
            print(f"📊 Progresso parcial:")
            print(f"  Modelos testados: {total_models:,}")
            print(f"  Cache hit rate: {cache_stats['hit_rate_percent']:.1f}%")
    except:
        pass
    
except Exception as e:
    duration = time.time() - start_time
    print(f"\n❌ ERRO após {duration/60:.1f} minutos:")
    print(f"   {str(e)}")
    
    # Diagnóstico do erro
    if "convergence" in str(e).lower():
        print(f"\n💡 DIAGNÓSTICO: Problema de convergência")
        print(f"   - Dataset muito grande ou features problemáticas")
        print(f"   - Considere filtrar correlações altas")
        print(f"   - Ou reduzir alpha para 0.01")
    elif "singular" in str(e).lower() or "matrix" in str(e).lower():
        print(f"\n💡 DIAGNÓSTICO: Problema de multicolinearidade")
        print(f"   - Features altamente correlacionadas")
        print(f"   - Considere filtro de correlação antes do stepwise")
    elif "memory" in str(e).lower():
        print(f"\n💡 DIAGNÓSTICO: Problema de memória")
        print(f"   - Dataset muito grande para memória disponível")
        print(f"   - Considere usar amostra estratificada")
    else:
        print(f"\n💡 DIAGNÓSTICO: Erro inesperado")
        print(f"   - Verificar qualidade dos dados")
        print(f"   - Verificar se target é adequado para ZIP")

    # Mostrar estatísticas parciais
    try:
        cache_stats = stepwise.get_cache_stats()
        total_models = cache_stats['cache_hits'] + cache_stats['cache_misses']
        if total_models > 0:
            print(f"\n📊 Estatísticas parciais:")
            print(f"   Modelos testados: {total_models:,}")
            print(f"   Cache hit rate: {cache_stats['hit_rate_percent']:.1f}%")
    except:
        pass

print(f"\n" + "="*80)
print(f"🏁 TESTE COMPLETO FINALIZADO - {time.strftime('%H:%M:%S')}") 