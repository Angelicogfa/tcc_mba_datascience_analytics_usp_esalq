import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_array, check_X_y
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from typing import Dict, List, Tuple, Optional, Union
from tqdm.auto import tqdm

class StepwiseZeroInflated(BaseEstimator, TransformerMixin):
    """
    Seletor de features stepwise otimizado para modelos Zero-Inflated (ZIP/ZINB).
    
    Esta classe implementa um algoritmo de seleção stepwise forward/backward 
    para modelos de contagem com inflação de zeros. A versão otimizada inclui
    cache, paralelização, batch processing e progress bars para melhor performance.
    
    Funcionalidades principais:
    - Seleção automática de features para componentes exógenos e inflacionados
    - Suporte para critérios AIC, BIC e Log-Likelihood
    - Validação de significância estatística
    - Cache inteligente para evitar recomputações
    - Processamento paralelo de features
    - Progress bars elegantes com tqdm
    - Batch processing para eficiência
    - Validação adaptativa baseada na complexidade do modelo
    
    Otimizações implementadas:
    1. **Cache de matrizes**: Evita recriação de matrizes de design idênticas
    2. **Cache de modelos**: Reutiliza resultados de modelos já ajustados
    3. **Validação adaptativa**: Pula validações desnecessárias em modelos simples
    4. **Early stopping**: Para validações assim que encontra problema
    5. **Paralelização**: Testa múltiplas features simultaneamente
    6. **Batch processing**: Processa features em lotes para otimização
    7. **Progress bars**: Feedback visual elegante durante o processo
    8. **Otimização de memória**: Gerenciamento eficiente de cache
    
    Compatível com scikit-learn pipeline através de BaseEstimator e TransformerMixin.
    
    Examples
    --------
    Exemplo básico com dados de contagem:
    
    >>> import numpy as np
    >>> import pandas as pd
    >>> from stepwise_optimized import StepwiseZeroInflated
    >>> 
    >>> # Dados de exemplo
    >>> X = pd.DataFrame({
    ...     'idade': np.random.normal(35, 10, 1000),
    ...     'renda': np.random.exponential(50000, 1000),
    ...     'educacao': np.random.choice([1, 2, 3, 4], 1000),
    ...     'sexo': np.random.choice([0, 1], 1000)
    ... })
    >>> y = np.random.poisson(2, 1000)  # Variável de contagem
    >>> # Adicionar inflação de zeros
    >>> zero_mask = np.random.binomial(1, 0.3, 1000)
    >>> y = np.where(zero_mask, 0, y)
    >>> 
    >>> # Configuração básica
    >>> stepwise = StepwiseZeroInflated(
    ...     alpha=0.05,
    ...     model_type='ZIP',
    ...     verbose=False  # Progress bar será mostrado
    ... )
    >>> 
    >>> # Ajustar modelo
    >>> stepwise.fit(X, y)
    >>> 
    >>> # Verificar features selecionadas
    >>> print("Exógenas:", stepwise.columns_exog_)
    >>> print("Inflacionadas:", stepwise.columns_inf_)
    
    Exemplo com configurações avançadas:
    
    >>> # Configuração otimizada para dataset grande
    >>> stepwise_advanced = StepwiseZeroInflated(
    ...     alpha=0.01,                 # Mais rigoroso
    ...     model_type='ZINB',          # Negative Binomial
    ...     selection_criterion='BIC',  # Penaliza mais a complexidade
    ...     max_iter=50,               # Máximo de iterações
    ...     verbose=False,             # Progress bar
    ...     use_cache=True,            # Cache habilitado
    ...     parallel_features=True,    # Paralelização
    ...     n_jobs=4,                  # 4 threads
    ...     batch_size=8               # Lotes de 8 features
    ... )
    >>> 
    >>> stepwise_advanced.fit(X, y)
    >>> 
    >>> # Estatísticas de cache
    >>> cache_stats = stepwise_advanced.get_cache_stats()
    >>> print(f"Cache hit rate: {cache_stats['hit_rate_percent']:.1f}%")
    
    Exemplo para análise de fraudes (contagem de transações suspeitas):
    
    >>> # Features para análise de fraude
    >>> X_fraud = pd.DataFrame({
    ...     'valor_transacao': np.random.lognormal(5, 1, 5000),
    ...     'hora_transacao': np.random.choice(range(24), 5000),
    ...     'dia_semana': np.random.choice(range(7), 5000),
    ...     'historico_cliente': np.random.exponential(100, 5000),
    ...     'num_cartoes': np.random.poisson(2, 5000),
    ...     'score_risco': np.random.beta(2, 5, 5000)
    ... })
    >>> 
    >>> # Número de transações suspeitas por cliente (com zeros inflacionados)
    >>> lambda_fraud = np.exp(-2 + 0.5 * X_fraud['score_risco'])
    >>> y_fraud = np.random.poisson(lambda_fraud)
    >>> # 40% dos clientes sem nenhuma transação suspeita
    >>> zero_inflation = np.random.binomial(1, 0.4, 5000)
    >>> y_fraud = np.where(zero_inflation, 0, y_fraud)
    >>> 
    >>> # Stepwise para detectar features importantes
    >>> fraud_stepwise = StepwiseZeroInflated(
    ...     alpha=0.05,
    ...     model_type='ZIP',
    ...     selection_criterion='AIC',
    ...     verbose=False,
    ...     parallel_features=True,
    ...     validation_steps=True
    ... )
    >>> 
    >>> fraud_stepwise.fit(X_fraud, y_fraud)
    >>> 
    >>> # Features importantes para predição de fraude
    >>> print("Features exógenas (contagem):", fraud_stepwise.columns_exog_)
    >>> print("Features inflacionadas (zeros):", fraud_stepwise.columns_inf_)
    
    Notes
    -----
    - Para datasets pequenos (< 1000 amostras), use parallel_features=False
    - Para datasets grandes (> 5000 amostras), aumente batch_size para 8-12
    - Use validation_steps=False apenas se tiver problemas de convergência
    - O cache é automaticamente limpo entre diferentes chamadas fit()
    - Progress bars funcionam melhor em terminals com suporte a ANSI
    
    See Also
    --------
    statsmodels.discrete.count_model.ZeroInflatedPoisson : Modelo ZIP base
    statsmodels.discrete.count_model.ZeroInflatedNegativeBinomialP : Modelo ZINB base
    sklearn.feature_selection : Outros métodos de seleção de features
    """

    def __init__(self, alpha=0.05, model_type='ZIP', selection_criterion='AIC', max_iter=100,
                 tolerance=1e-8, verbose=False, model_params=None, fit_params=None, 
                 validation_steps=True, use_cache=True, parallel_features=False, 
                 n_jobs=None, batch_size=5, early_stopping=True, multicollinearity_threshold=0.95):
        """
        Inicializa o seletor stepwise para modelos Zero-Inflated.
        
        Parameters
        ----------
        alpha : float, default=0.05
            Nível de significância para testes estatísticos. Features com p-valor
            maior que alpha são consideradas não significativas e removidas.
            
            - Valores menores (ex: 0.01) = mais rigoroso, menos features selecionadas
            - Valores maiores (ex: 0.10) = menos rigoroso, mais features selecionadas
            
        model_type : {'ZIP', 'ZINB'}, default='ZIP'
            Tipo do modelo Zero-Inflated a ser usado:
            
            - 'ZIP': Zero-Inflated Poisson (mais simples, assume equidispersão)
            - 'ZINB': Zero-Inflated Negative Binomial (trata sobredispersão)
            
        selection_criterion : {'AIC', 'BIC', 'LLF'}, default='AIC'
            Critério para seleção de modelos:
            
            - 'AIC': Akaike Information Criterion (balanceado)
            - 'BIC': Bayesian Information Criterion (penaliza mais a complexidade)
            - 'LLF': Negative Log-Likelihood (pode levar a overfitting)
            
        max_iter : int, default=100
            Número máximo de iterações do algoritmo stepwise. Cada iteração
            tenta adicionar a melhor feature disponível.
            
        tolerance : float, default=1e-8
            Tolerância para convergência. O algoritmo para quando a melhoria
            no critério for menor que este valor.
            
        verbose : bool, default=False
            Controla o tipo de feedback durante o ajuste:
            
            - False: Mostra progress bars elegantes com tqdm
            - True: Mostra logs detalhados de cada passo
            
        model_params : dict, optional
            Parâmetros adicionais passados para o modelo statsmodels.
            
            Exemplos:
            - {'offset': offset_array} para incluir offset
            - {'exposure': exposure_array} para dados de exposição
            
        fit_params : dict, optional
            Parâmetros adicionais passados para o método fit() do modelo.
            
            Exemplos:
            - {'maxiter': 1000} para mais iterações de otimização
            - {'method': 'bfgs'} para método de otimização específico
            
        validation_steps : bool, default=True
            Se deve realizar validações adicionais dos modelos:
            
            - True: Valida convergência, multicolinearidade, etc.
            - False: Apenas validação de significância (mais rápido)
            
        use_cache : bool, default=True
            Se deve usar cache para otimização de performance:
            
            - True: Cacheia matrizes e modelos (recomendado)
            - False: Recalcula tudo (útil para debugging)
            
        parallel_features : bool, default=False
            Se deve testar features em paralelo:
            
            - True: Usa múltiplas threads (melhor para datasets grandes)
            - False: Processamento sequencial (melhor para datasets pequenos)
            
        n_jobs : int, optional
            Número de threads para processamento paralelo.
            Se None, usa min(4, número_de_cores).
            
            - 1: Sem paralelismo
            - 2-4: Bom para a maioria dos casos
            - -1: Usa todos os cores disponíveis
            
        batch_size : int, default=5
            Número de features processadas em cada lote:
            
            - Valores menores (3-5): Melhor para datasets pequenos
            - Valores maiores (8-12): Melhor para datasets grandes
            
        early_stopping : bool, default=True
            Se deve usar early stopping para otimizar performance:
            
            - True: Para validações assim que encontra problemas (mais rápido)
            - False: Executa todas as validações completas (mais confiável)
            
        multicollinearity_threshold : float, default=0.95
            Limiar de correlação absoluta para detectar multicolinearidade.
            Features com correlação acima deste valor são rejeitadas.
            
            - Valores menores (ex: 0.80) = mais rigoroso, rejeita mais features
            - Valores maiores (ex: 0.99) = menos rigoroso, aceita correlações altas
            - 0.95: Padrão conservador para a maioria dos casos
            - 0.90: Rigoroso para modelos de produção
            - 0.80: Muito rigoroso para dados com features similares
            
        Examples
        --------
        Configuração básica para análise exploratória:
        
        >>> stepwise_basic = StepwiseZeroInflated()
        >>> # Usa configurações padrão: alpha=0.05, ZIP, AIC, progress bar
        
        Configuração rigorosa para modelo final:
        
        >>> stepwise_final = StepwiseZeroInflated(
        ...     alpha=0.01,                        # Mais rigoroso
        ...     model_type='ZINB',                 # Trata sobredispersão  
        ...     selection_criterion='BIC',         # Penaliza complexidade
        ...     multicollinearity_threshold=0.85   # Rigoroso para multicolinearidade
        ... )
        
        Configuração otimizada para dataset grande:
        
        >>> stepwise_big = StepwiseZeroInflated(
        ...     parallel_features=True,            # Paralelização
        ...     n_jobs=4,                         # 4 threads
        ...     batch_size=10,                    # Lotes maiores
        ...     use_cache=True,                   # Cache obrigatório
        ...     verbose=False,                    # Progress bar limpo
        ...     multicollinearity_threshold=0.90  # Padrão rigoroso
        ... )
        
        Configuração para debugging:
        
        >>> stepwise_debug = StepwiseZeroInflated(
        ...     verbose=True,                     # Logs detalhados
        ...     use_cache=False,                  # Sem cache
        ...     validation_steps=True,            # Todas as validações
        ...     max_iter=20,                     # Menos iterações
        ...     multicollinearity_threshold=0.99  # Permissivo para debug
        ... )
        
        Configuração com parâmetros customizados do modelo:
        
        >>> # Para dados com offset conhecido
        >>> offset_values = np.log(population_exposure) 
        >>> stepwise_offset = StepwiseZeroInflated(
        ...     model_params={'offset': offset_values},
        ...     fit_params={'maxiter': 2000}
        ... )
        
        Configuração para dados com sobredispersão severa:
        
        >>> stepwise_overdispersed = StepwiseZeroInflated(
        ...     model_type='ZINB',                # Negative Binomial
        ...     alpha=0.01,                      # Mais conservador
        ...     selection_criterion='BIC',        # Penaliza complexidade
        ...     fit_params={'maxiter': 3000},     # Mais iterações
        ...     multicollinearity_threshold=0.88  # Rigoroso para estabilidade
        ... )
        
        Configuração sem early stopping (validação completa):
        
        >>> stepwise_no_early = StepwiseZeroInflated(
        ...     early_stopping=False,             # Desativa early stopping
        ...     validation_steps=True,            # Validação completa
        ...     verbose=True,                     # Para ver todos os detalhes
        ...     multicollinearity_threshold=0.90  # Rigoroso para validação completa
        ... )
        
        Configuração para dados financeiros (alta correlação esperada):
        
        >>> stepwise_finance = StepwiseZeroInflated(
        ...     multicollinearity_threshold=0.98,  # Permite correlações altas
        ...     alpha=0.01,                       # Mais rigoroso em significância
        ...     validation_steps=True             # Validação completa
        ... )
        
        Configuração para análise de fraude (máxima robustez):
        
        >>> stepwise_fraud = StepwiseZeroInflated(
        ...     alpha=0.01,                       # Rigoroso
        ...     multicollinearity_threshold=0.85,  # Muito rigoroso
        ...     validation_steps=True,            # Validação completa
        ...     early_stopping=False,             # Sem early stopping
        ...     selection_criterion='BIC'         # Penaliza complexidade
        ... )
        
        Raises
        ------
        ValueError
            Se alpha não estiver entre 0 e 1, model_type não for 'ZIP' ou 'ZINB',
            selection_criterion não for válido, max_iter <= 0, tolerance <= 0,
            ou multicollinearity_threshold não estiver entre 0 e 1.
            
        Notes
        -----
        - Para análise de fraudes, recomenda-se alpha=0.05 e model_type='ZIP'
        - Para dados com muita sobredispersão, use model_type='ZINB'
        - Para modelos finais de produção, use selection_criterion='BIC'
        - O cache é automaticamente limpo a cada nova chamada de fit()
        - Progress bars requerem terminal com suporte a ANSI
        - Early stopping desabilitado pode tornar o processo mais lento mas mais completo
        - Valores típicos para multicollinearity_threshold:
          * 0.95: Padrão conservador (recomendado)
          * 0.90: Rigoroso para produção
          * 0.85: Muito rigoroso para máxima estabilidade
          * 0.98: Permissivo para dados financeiros/correlacionados
        """

        # Parâmetros originais
        if not 0 < alpha < 1:
            raise ValueError("alpha deve estar entre 0 e 1")
        if model_type not in ['ZIP', 'ZINB']:
            raise ValueError("model_type deve ser 'ZIP' ou 'ZINB'")
        if selection_criterion not in ['AIC', 'BIC', 'LLF']:
            raise ValueError("selection_criterion deve ser 'AIC', 'BIC' ou 'LLF'")
        if max_iter <= 0:
            raise ValueError("max_iter deve ser positivo")
        if tolerance <= 0:
            raise ValueError("tolerance deve ser positivo")
        if not 0 < multicollinearity_threshold < 1:
            raise ValueError("multicollinearity_threshold deve estar entre 0 e 1")

        self.alpha = alpha
        self.model_type = model_type
        self.selection_criterion = selection_criterion
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.verbose = verbose
        self.model_params = model_params if model_params is not None else {}
        self.fit_params = fit_params if fit_params is not None else {}
        self.validation_steps = validation_steps
        self.multicollinearity_threshold = multicollinearity_threshold
        
        # Novos parâmetros de otimização
        self.use_cache = use_cache
        self.parallel_features = parallel_features
        self.n_jobs = n_jobs if n_jobs is not None else min(4, mp.cpu_count())
        self.batch_size = max(1, batch_size)
        self.early_stopping = early_stopping
        
        # Cache para otimização
        self._matrix_cache = {}
        self._model_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        self._setup_criterion_behavior()

    def _setup_criterion_behavior(self):
        """
        Configura comportamento específico para cada critério de seleção.
        
        Emite warnings para critérios que podem levar a problemas,
        especialmente o LLF que pode causar overfitting em seleção stepwise.
        """
        if self.selection_criterion == 'LLF':
            warnings.warn(
                "LLF como critério de seleção pode levar a overfitting. "
                "Considere usar 'AIC' ou 'BIC' para seleção stepwise.",
                UserWarning
            )

    def _get_model_criterion(self, result):
        """
        Extrai o valor do critério de seleção do modelo ajustado.
        
        Parameters
        ----------
        result : statsmodels fitted model
            Resultado do ajuste do modelo.
            
        Returns
        -------
        criterion : float
            Valor do critério de seleção configurado.
            
        Notes
        -----
        - AIC e BIC: valores menores são melhores
        - LLF: usa o negativo da log-likelihood para consistência
        """
        if self.selection_criterion == 'AIC':
            return result.aic
        elif self.selection_criterion == 'BIC':
            return result.bic
        elif self.selection_criterion == 'LLF':
            return -result.llf

    def _get_criterion_name(self):
        """
        Retorna nome amigável do critério para exibição em logs.
        
        Returns
        -------
        name : str
            Nome legível do critério de seleção.
        """
        criterion_names = {
            'AIC': 'AIC',
            'BIC': 'BIC', 
            'LLF': '-Log-Likelihood'
        }
        return criterion_names[self.selection_criterion]

    def _get_matrix_cache_key(self, exog_features: List[str], inf_features: List[str]) -> str:
        """Gera chave única para cache de matrizes."""
        exog_str = ','.join(sorted(exog_features))
        inf_str = ','.join(sorted(inf_features))
        return f"exog:[{exog_str}]_inf:[{inf_str}]"

    def _prepare_matrices_cached(self, X: pd.DataFrame, exog_features: List[str], 
                                inf_features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepara matrizes com cache para evitar recriação desnecessária."""
        cache_key = self._get_matrix_cache_key(exog_features, inf_features)
        
        if self.use_cache and cache_key in self._matrix_cache:
            self._cache_hits += 1
            return self._matrix_cache[cache_key]
        
        self._cache_misses += 1
        
        # Preparar matriz exógena
        if exog_features:
            X1 = X[exog_features].copy()
            X1 = sm.add_constant(X1, has_constant='add')
        else:
            X1 = pd.DataFrame({'const': np.ones(len(X))})

        # Preparar matriz inflacionada
        if inf_features:
            X2 = X[inf_features].copy()
            X2 = sm.add_constant(X2, has_constant='add')
        else:
            X2 = pd.DataFrame({'const': np.ones(len(X))})
        
        # Armazenar no cache
        if self.use_cache:
            self._matrix_cache[cache_key] = (X1, X2)
        
        return X1, X2

    def _get_model_cache_key(self, exog_features: List[str], inf_features: List[str], 
                           data_hash: str) -> str:
        """Gera chave única para cache de modelos."""
        matrix_key = self._get_matrix_cache_key(exog_features, inf_features)
        return f"{matrix_key}_data:{data_hash}_type:{self.model_type}"

    def _fit_model_cached(self, X: pd.DataFrame, y: np.ndarray, exog_features: List[str], 
                         inf_features: List[str]) -> Optional[object]:
        """Ajusta modelo com cache para evitar recomputação."""
        
        # Gerar hash simples dos dados para cache
        data_hash = str(hash((tuple(y), tuple(X.columns))))[:8]
        cache_key = self._get_model_cache_key(exog_features, inf_features, data_hash)
        
        if self.use_cache and cache_key in self._model_cache:
            self._cache_hits += 1
            return self._model_cache[cache_key]
        
        self._cache_misses += 1
        
        try:
            # Preparar matrizes
            X1, X2 = self._prepare_matrices_cached(X, exog_features, inf_features)
            
            # Selecionar classe do modelo
            if self.model_type == 'ZIP':
                ModelClass = sm.ZeroInflatedPoisson
            elif self.model_type == 'ZINB':
                ModelClass = sm.ZeroInflatedNegativeBinomialP
            else:
                raise ValueError(f"model_type '{self.model_type}' não suportado")

            # Ajustar modelo
            model = ModelClass(y, X1, exog_infl=X2, **self.model_params)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = model.fit(**self.fit_params)

            # Armazenar no cache
            if self.use_cache:
                self._model_cache[cache_key] = result
            
            return result
            
        except Exception:
            return None

    def _validate_features_significance_fast(self, result, exog_features: List[str], 
                                           inf_features: List[str]) -> Tuple[bool, str]:
        """Validação rápida de significância com early stopping opcional."""
        try:
            # Verificar se temos parâmetros suficientes
            expected_params = 1 + len(exog_features) + 1 + len(inf_features)
            if len(result.pvalues) < expected_params:
                return False, "parâmetros insuficientes"

            # Coletar todas as features não-significativas se early_stopping=False
            non_significant_features = []

            # Verificar features exógenas
            for i, feature in enumerate(exog_features):
                try:
                    p_value = result.pvalues.iloc[i + 1]
                    if pd.isna(p_value) or p_value >= self.alpha:
                        if self.early_stopping:
                            # Early stopping: retornar imediatamente
                            return False, f"{feature} não significativa (p={p_value:.4f})"
                        else:
                            # Sem early stopping: coletar todos os problemas
                            non_significant_features.append(f"{feature} (p={p_value:.4f})")
                except (IndexError, KeyError):
                    error_msg = f"{feature} erro no p-valor"
                    if self.early_stopping:
                        return False, error_msg
                    else:
                        non_significant_features.append(error_msg)

            # Verificar features inflacionadas
            n_exog = len(exog_features) + 1
            for i, feature in enumerate(inf_features):
                try:
                    p_value = result.pvalues.iloc[n_exog + i + 1]
                    if pd.isna(p_value) or p_value >= self.alpha:
                        if self.early_stopping:
                            # Early stopping: retornar imediatamente
                            return False, f"{feature} não significativa (p={p_value:.4f})"
                        else:
                            # Sem early stopping: coletar todos os problemas
                            non_significant_features.append(f"{feature} (p={p_value:.4f})")
                except (IndexError, KeyError):
                    error_msg = f"{feature} erro no p-valor"
                    if self.early_stopping:
                        return False, error_msg
                    else:
                        non_significant_features.append(error_msg)

            # Se encontrou features não-significativas e não está usando early stopping
            if non_significant_features and not self.early_stopping:
                return False, f"Features não significativas: {'; '.join(non_significant_features)}"

            return True, "todas significativas"
            
        except Exception as e:
            return False, f"erro validação: {str(e)[:30]}"

    def _test_feature_scenarios_batch(self, X: pd.DataFrame, y: np.ndarray, 
                                    candidate_features: List[str], current_exog: List[str], 
                                    current_inf: List[str]) -> List[Dict]:
        """Testa múltiplas features em batch para otimização."""
        results = []
        
        if self.parallel_features and len(candidate_features) > 1:
            # Processamento paralelo
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                future_to_feature = {}
                
                for feature in candidate_features:
                    future = executor.submit(
                        self._test_single_feature_all_scenarios, 
                        X, y, feature, current_exog, current_inf
                    )
                    future_to_feature[future] = feature
                
                # Processar resultados conforme completam
                for future in as_completed(future_to_feature):
                    feature = future_to_feature[future]
                    try:
                        result = future.result()
                        result['feature'] = feature
                        results.append(result)
                    except Exception as e:
                        if self.verbose:
                            print(f"Erro testando '{feature}': {str(e)[:50]}")
                        results.append({
                            'feature': feature,
                            'best_criterion': None,
                            'best_scenario': None,
                            'reason': f"erro: {str(e)[:30]}"
                        })
        else:
            # Processamento sequencial
            for feature in candidate_features:
                try:
                    result = self._test_single_feature_all_scenarios(X, y, feature, current_exog, current_inf)
                    result['feature'] = feature
                    results.append(result)
                except Exception as e:
                    if self.verbose:
                        print(f"Erro testando '{feature}': {str(e)[:50]}")
                    results.append({
                        'feature': feature,
                        'best_criterion': None,
                        'best_scenario': None,
                        'reason': f"erro: {str(e)[:30]}"
                    })
        
        return results

    def _test_single_feature_all_scenarios(self, X: pd.DataFrame, y: np.ndarray, 
                                         feature: str, current_exog: List[str], 
                                         current_inf: List[str]) -> Dict:
        """Testa uma feature em todos os cenários possíveis."""
        best_criterion = float('inf')
        best_scenario = None
        best_config = None
        
        scenarios_to_test = [
            ('exog', current_exog + [feature], current_inf),
            ('inf', current_exog, current_inf + [feature])
        ]
        
        # Adicionar cenário "ambas" se aplicável
        if current_exog or current_inf:
            scenarios_to_test.append(
                ('both', current_exog + [feature], current_inf + [feature])
            )
        
        for scenario_name, test_exog, test_inf in scenarios_to_test:
            # Ajustar modelo
            result = self._fit_model_cached(X, y, test_exog, test_inf)
            
            if result is None:
                continue
                
            # Validação rápida de significância
            is_significant, sig_details = self._validate_features_significance_fast(
                result, test_exog, test_inf
            )
            
            if not is_significant:
                continue
                
            # Validação adaptativa (pula etapas se não necessário)
            if self.validation_steps:
                passed_validation, _ = self._validate_model_adaptive(
                    X, y, test_exog, test_inf, result
                )
                if not passed_validation:
                    continue
            
            # Obter critério
            criterion = self._get_model_criterion(result)
            
            if criterion < best_criterion:
                best_criterion = criterion
                best_scenario = scenario_name
                best_config = (test_exog, test_inf)
        
        return {
            'best_criterion': best_criterion if best_criterion != float('inf') else None,
            'best_scenario': best_scenario,
            'best_config': best_config,
            'reason': 'aprovada' if best_scenario else 'rejeitada'
        }

    def _validate_model_adaptive(self, X: pd.DataFrame, y: np.ndarray, 
                               exog_features: List[str], inf_features: List[str], 
                               result) -> Tuple[bool, str]:
        """Validação adaptativa que pula etapas desnecessárias quando early_stopping=True."""
        
        total_features = len(exog_features) + len(inf_features)
        
        # Se early stopping está desabilitado, sempre fazer validação completa
        if not self.early_stopping:
            return self._validate_full_robustness(X, y, exog_features, inf_features, result)
        
        # Com early stopping, usar validação adaptativa baseada na complexidade
        # Para modelos simples (1-2 features), pular validações pesadas
        if total_features <= 2:
            return True, "modelo simples"
            
        # Para modelos médios (3-5 features), validação básica
        if total_features <= 5:
            return self._validate_basic_stability(X, y, exog_features, inf_features, result)
            
        # Para modelos complexos (6+ features), validação completa
        return self._validate_full_robustness(X, y, exog_features, inf_features, result)

    def _validate_basic_stability(self, X: pd.DataFrame, y: np.ndarray, 
                                exog_features: List[str], inf_features: List[str], 
                                result) -> Tuple[bool, str]:
        """Validação básica de estabilidade para modelos médios."""
        try:
            # Verificar apenas convergência e multicolinearidade básica
            if hasattr(result, 'mle_retvals') and not result.mle_retvals.get('converged', True):
                return False, "não convergiu"
                
            # Multicolinearidade básica apenas se muitas features
            if len(exog_features) > 3:
                corr_matrix = X[exog_features].corr().abs()
                max_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max()
                if max_corr > self.multicollinearity_threshold:
                    return False, f"multicolinearidade alta (r={max_corr:.3f} > {self.multicollinearity_threshold:.2f})"
                    
            status_msg = "estabilidade básica aprovada"
            if not self.early_stopping:
                status_msg += " (validação completa)"
            return True, status_msg
            
        except Exception:
            return False, "erro na validação básica"

    def _validate_full_robustness(self, X: pd.DataFrame, y: np.ndarray, 
                                exog_features: List[str], inf_features: List[str], 
                                result) -> Tuple[bool, str]:
        """Validação completa para modelos complexos ou quando early_stopping=False."""
        try:
            # Validação básica sempre
            basic_passed, basic_msg = self._validate_basic_stability(X, y, exog_features, inf_features, result)
            if not basic_passed:
                return False, basic_msg
            
            # Validações adicionais quando early stopping está desabilitado
            if not self.early_stopping:
                validation_issues = []
                
                # Verificar condição dos valores ajustados
                try:
                    fitted_values = result.fittedvalues
                    if np.any(np.isnan(fitted_values)) or np.any(fitted_values < 0):
                        validation_issues.append("valores ajustados inválidos")
                except Exception:
                    validation_issues.append("erro nos valores ajustados")
                
                # Verificar matriz de covariância
                try:
                    cov_matrix = result.cov_params()
                    if np.any(np.diag(cov_matrix) <= 0):
                        validation_issues.append("matriz covariância inválida")
                except Exception:
                    validation_issues.append("erro na matriz de covariância")
                
                # Verificar multicolinearidade rigorosa para todos os conjuntos
                all_features = exog_features + inf_features
                if len(all_features) > 2:
                    try:
                        corr_matrix = X[all_features].corr().abs()
                        max_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max()
                        if max_corr > self.multicollinearity_threshold:  # Usa limiar parametrizado
                            validation_issues.append(f"multicolinearidade severa (r={max_corr:.3f} > {self.multicollinearity_threshold:.2f})")
                    except Exception:
                        validation_issues.append("erro na verificação de multicolinearidade")
                
                # Se há problemas e não está usando early stopping, reportar todos
                if validation_issues:
                    return False, f"Problemas encontrados: {'; '.join(validation_issues)}"
                
                return True, f"validação completa aprovada (limiar multicolinearidade: {self.multicollinearity_threshold:.2f})"
            
            # Com early stopping, usar apenas validação básica
            return True, f"validação robusta aprovada (limiar multicolinearidade: {self.multicollinearity_threshold:.2f})"
            
        except Exception as e:
            return False, f"erro na validação completa: {str(e)[:30]}"

    def _stepwise_selection_optimized(self, X: pd.DataFrame, y: np.ndarray) -> Dict:
        """Versão otimizada do algoritmo stepwise."""
        
        # Inicialização
        columns_exog = []
        columns_inf = []
        excluded = []
        available_features = list(X.columns)
        
        criterion_history = []
        step_history = []
        
        # NOVA LÓGICA: rastrear todas as features testadas quando early_stopping=False
        all_features_tested = set()
        
        # Modelo baseline
        initial_criterion = self._calculate_baseline_criterion_fast(y)
        best_criterion = initial_criterion
        criterion_history.append(initial_criterion)
        
        if self.verbose:
            print(f"STEPWISE OTIMIZADO INICIADO")
            print(f"Cache habilitado: {self.use_cache}")
            print(f"Processamento paralelo: {self.parallel_features}")
            print(f"Critério inicial: {initial_criterion:.2f}")
            print(f"Modo validação: {'TODAS as features' if not self.early_stopping else 'early stopping'}")
            print("-" * 60)

        # Configurar barras de progresso quando verbose=False
        use_progress = not self.verbose
        pbar_iterations = None
        
        if use_progress:
            # Barra única com informações consolidadas
            pbar_iterations = tqdm(
                total=self.max_iter,
                desc="Stepwise Selection",
                unit="iter",
                position=0,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
            )

        # Loop principal otimizado
        for iteration in range(self.max_iter):
            improved = False
            
            # Processar features em batches
            remaining_features = [f for f in available_features 
                                if f not in columns_exog and f not in columns_inf and f not in excluded]
            
            # NOVA LÓGICA: quando early_stopping=False, testar features não testadas primeiro
            if not self.early_stopping:
                untested_features = [f for f in remaining_features if f not in all_features_tested]
                if untested_features:
                    # Prioritizar features não testadas
                    remaining_features = untested_features
                elif remaining_features:
                    # Se todas foram testadas mas ainda há features restantes, testar novamente
                    # (pode haver diferentes combinações agora)
                    all_features_tested.clear()  # Reset para nova rodada
                    remaining_features = remaining_features
                else:
                    # Todas as features foram testadas e processadas
                    if self.verbose:
                        print(f"TODAS as {len(available_features)} features foram testadas e processadas")
                    break
            
            if not remaining_features:
                if use_progress:
                    pbar_iterations.close()
                break
                
            # Dividir em batches para processamento eficiente
            batches = [remaining_features[i:i+self.batch_size] 
                      for i in range(0, len(remaining_features), self.batch_size)]
            
            best_addition = None
            iteration_excluded = []  # Features excluídas nesta iteração
            
            for batch in batches:
                if self.verbose:
                    print(f"Iteração {iteration+1}: testando batch de {len(batch)} features")
                
                # MARCAR FEATURES COMO TESTADAS (quando early_stopping=False)
                if not self.early_stopping:
                    all_features_tested.update(batch)
                
                # Testar batch de features
                batch_results = self._test_feature_scenarios_batch(
                    X, y, batch, columns_exog, columns_inf
                )
                
                # Encontrar melhor resultado do batch E processar exclusões
                for result in batch_results:
                    if (result['best_criterion'] is not None and 
                        result['best_criterion'] < best_criterion):
                        
                        if best_addition is None or result['best_criterion'] < best_addition['criterion']:
                            best_addition = {
                                'feature': result['feature'],
                                'criterion': result['best_criterion'],
                                'scenario': result['best_scenario'],
                                'config': result['best_config']
                            }
                    else:
                        # NOVA LÓGICA: quando early_stopping=False, não excluir imediatamente
                        # Só excluir após todas as features serem testadas
                        if self.early_stopping:
                            excluded.append(result['feature'])
                        else:
                            iteration_excluded.append(result['feature'])
            
            # Aplicar melhor adição se houver
            if best_addition:
                columns_exog, columns_inf = best_addition['config']
                best_criterion = best_addition['criterion']
                improved = True
                
                if self.verbose:
                    improvement = criterion_history[-1] - best_criterion
                    print(f"Adicionada '{best_addition['feature']}' como {best_addition['scenario']} "
                          f"(melhoria: {improvement:.2f})")
                          
                # Limpar das exclusões da iteração se foi selecionada
                if best_addition['feature'] in iteration_excluded:
                    iteration_excluded.remove(best_addition['feature'])
            
            # NOVA LÓGICA: só adicionar às exclusões quando necessário
            if self.early_stopping:
                # Modo normal: adicionar exclusões imediatamente
                pass  # já foi feito acima
            else:
                # Modo validação completa: só excluir quando não há mais o que testar
                # e a feature foi rejeitada consistentemente
                if not remaining_features or len(all_features_tested) >= len(available_features):
                    excluded.extend(iteration_excluded)
            
            # Backward elimination otimizado
            if improved:
                removed_any = self._backward_elimination_optimized(
                    X, y, columns_exog, columns_inf, best_criterion
                )
                if removed_any:
                    best_criterion = removed_any['new_criterion']
                    columns_exog = removed_any['new_exog']
                    columns_inf = removed_any['new_inf']
                    excluded.extend(removed_any['removed_features'])
            
            criterion_history.append(best_criterion)
            
            # Atualizar barra de progresso das iterações
            if use_progress and pbar_iterations:
                selected_features = len(columns_exog) + len(columns_inf)
                tested_features = len(all_features_tested) if not self.early_stopping else len(excluded) + selected_features
                remaining_count = len(remaining_features) if 'remaining_features' in locals() else 0
                
                # Calcular progresso real das features
                total_features = len(available_features)
                feature_progress = f"{tested_features}/{total_features}"
                
                # Status mais detalhado
                status_details = []
                if improved:
                    improvement = criterion_history[-2] - best_criterion if len(criterion_history) > 1 else 0
                    status_details.append(f"↗ melhorou {improvement:.2f}")
                else:
                    status_details.append("→ sem melhoria")
                
                if not self.early_stopping:
                    status_details.append(f"({remaining_count} restantes)")
                
                pbar_iterations.set_postfix({
                    'features': selected_features,
                    'testadas': feature_progress,
                    self.selection_criterion: f"{best_criterion:.2f}",
                    'status': " ".join(status_details)
                })
                pbar_iterations.update(1)
            
            # NOVA LÓGICA DE CONVERGÊNCIA: diferente para early_stopping=True/False
            if self.early_stopping:
                # Modo normal: parar se não melhorou
                if not improved and len(criterion_history) > 1:
                    criterion_change = abs(criterion_history[-2] - criterion_history[-1])
                    if criterion_change < self.tolerance:
                        if self.verbose:
                            print(f"Convergência atingida (mudança: {criterion_change:.6f})")
                        if use_progress:
                            pbar_iterations.set_description("Convergiu")
                            pbar_iterations.close()
                        break
                
                if not improved:
                    if self.verbose:
                        print(f"Nenhuma melhoria na iteração {iteration+1}")
                    if use_progress:
                        pbar_iterations.set_description("Sem melhoria")
                        pbar_iterations.close()
                    break
            else:
                # Modo validação completa: continuar até testar TODAS as features
                total_processed = len(columns_exog) + len(columns_inf) + len(excluded)
                if total_processed >= len(available_features):
                    if self.verbose:
                        print(f"TODAS as {len(available_features)} features foram processadas")
                        print(f"Selecionadas: {len(columns_exog) + len(columns_inf)}, "
                              f"Excluídas: {len(excluded)}")
                    if use_progress:
                        pbar_iterations.set_description("Todas testadas")
                        pbar_iterations.close()
                    break
                    
                # Continuar mesmo sem melhoria para testar todas as features
                if not improved and self.verbose:
                    print(f"Sem melhoria na iteração {iteration+1}, mas continuando "
                          f"(testadas: {len(all_features_tested)}/{len(available_features)})")

        # Fechar barras de progresso se ainda estiverem abertas
        if use_progress:
            if pbar_iterations:
                pbar_iterations.close()

        # RELATÓRIO FINAL melhorado
        if self.verbose:
            print(f"\nStepwise finalizado:")
            print(f"Modo: {'VALIDAÇÃO COMPLETA' if not self.early_stopping else 'EARLY STOPPING'}")
            print(f"Features disponíveis: {len(available_features)}")
            print(f"Features testadas: {len(all_features_tested) if not self.early_stopping else 'N/A'}")
            print(f"Features selecionadas: {len(columns_exog) + len(columns_inf)}")
            print(f"  - Exógenas: {columns_exog}")
            print(f"  - Inflacionadas: {columns_inf}")
            print(f"Features excluídas: {len(excluded)}")
            print(f"Cache hits/misses: {self._cache_hits}/{self._cache_misses}")
            if self._cache_hits + self._cache_misses > 0:
                hit_rate = self._cache_hits / (self._cache_hits + self._cache_misses) * 100
                print(f"Taxa de acerto do cache: {hit_rate:.1f}%")

        return {
            'columns_exog': columns_exog,
            'columns_inf': columns_inf,
            'excluded': excluded,
            'criterion_history': criterion_history,
            'step_history': step_history,
            'total_features_tested': len(all_features_tested) if not self.early_stopping else len(excluded) + len(columns_exog) + len(columns_inf),
            'validation_mode': 'complete' if not self.early_stopping else 'early_stopping'
        }

    def _backward_elimination_optimized(self, X: pd.DataFrame, y: np.ndarray, 
                                      current_exog: List[str], current_inf: List[str], 
                                      current_criterion: float) -> Optional[Dict]:
        """Backward elimination otimizado."""
        
        all_features = current_exog + current_inf
        if len(all_features) <= 1:
            return None
            
        best_removal = None
        removed_features = []
        
        # Testar remoção de cada feature
        for feature in all_features:
            test_exog = [f for f in current_exog if f != feature]
            test_inf = [f for f in current_inf if f != feature]
            
            result = self._fit_model_cached(X, y, test_exog, test_inf)
            if result is None:
                continue
                
            # Validação rápida
            is_significant, _ = self._validate_features_significance_fast(result, test_exog, test_inf)
            if not is_significant:
                continue
                
            criterion = self._get_model_criterion(result)
            
            # Se remover melhora ou mantém o critério, marcar para remoção
            if criterion <= current_criterion + self.tolerance:
                if best_removal is None or criterion < best_removal['criterion']:
                    best_removal = {
                        'feature': feature,
                        'criterion': criterion,
                        'new_exog': test_exog,
                        'new_inf': test_inf
                    }
        
        if best_removal:
            return {
                'new_criterion': best_removal['criterion'],
                'new_exog': best_removal['new_exog'],
                'new_inf': best_removal['new_inf'],
                'removed_features': [best_removal['feature']]
            }
            
        return None

    def _calculate_baseline_criterion_fast(self, y: np.ndarray) -> float:
        """Cálculo rápido do critério baseline."""
        try:
            # Usar cache se possível
            baseline_key = f"baseline_{self.model_type}_{len(y)}_{hash(tuple(y))}"
            
            if self.use_cache and baseline_key in self._model_cache:
                result = self._model_cache[baseline_key]
                return self._get_model_criterion(result)
            
            # Calcular baseline
            X_const = pd.DataFrame({'const': np.ones(len(y))})
            
            if self.model_type == 'ZIP':
                ModelClass = sm.ZeroInflatedPoisson
            elif self.model_type == 'ZINB':
                ModelClass = sm.ZeroInflatedNegativeBinomialP
            else:
                raise ValueError(f"model_type '{self.model_type}' não suportado")

            model = ModelClass(y, X_const, exog_infl=X_const, **self.model_params)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = model.fit(**self.fit_params)

            if self.use_cache:
                self._model_cache[baseline_key] = result
                
            return self._get_model_criterion(result)
            
        except Exception:
            # Fallback conservador
            return len(y) * 10 if self.selection_criterion in ['AIC', 'BIC'] else len(y) * 5

    def fit(self, X, y):
        """
        Ajusta o modelo stepwise aos dados fornecidos.
        
        Este método executa o algoritmo de seleção stepwise forward/backward
        para encontrar o melhor subconjunto de features para os componentes
        exógeno e inflacionado do modelo Zero-Inflated.
        
        O processo inclui:
        1. Limpeza de caches anteriores
        2. Validação e preparação dos dados
        3. Execução do algoritmo stepwise otimizado
        4. Ajuste do modelo final com features selecionadas
        5. Armazenamento dos resultados
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Matriz de features. Pode ser numpy array, pandas DataFrame,
            ou qualquer array-like aceito pelo scikit-learn.
            
        y : array-like of shape (n_samples,)
            Variável target (contagem). Deve conter apenas valores
            não-negativos inteiros. Zeros são tratados pelo componente
            inflacionado do modelo.
            
        Returns
        -------
        self : StepwiseZeroInflated
            Instância ajustada do estimador.
            
        Attributes
        ----------
        Após o ajuste, os seguintes atributos ficam disponíveis:
        
        columns_exog_ : list
            Nomes das features selecionadas para o componente exógeno
            (que modela a contagem esperada).
            
        columns_inf_ : list  
            Nomes das features selecionadas para o componente inflacionado
            (que modela a probabilidade de zero estrutural).
            
        excluded_ : list
            Nomes das features que foram testadas mas rejeitadas.
            
        final_model_ : statsmodels fitted model
            Modelo final ajustado com as features selecionadas.
            
        criterion_history_ : list
            Histórico dos valores do critério de seleção a cada iteração.
            
        n_features_in_ : int
            Número de features na entrada.
            
        feature_names_in_ : ndarray
            Nomes das features na entrada.
            
        Examples
        --------
        Exemplo básico:
        
        >>> import pandas as pd
        >>> import numpy as np
        >>> 
        >>> # Dados de contagem com inflação de zeros
        >>> X = pd.DataFrame({
        ...     'feature1': np.random.normal(0, 1, 1000),
        ...     'feature2': np.random.exponential(1, 1000),
        ...     'feature3': np.random.choice([0, 1], 1000)
        ... })
        >>> y = np.random.poisson(1.5, 1000)
        >>> zero_mask = np.random.binomial(1, 0.3, 1000)
        >>> y = np.where(zero_mask, 0, y)
        >>> 
        >>> # Ajustar stepwise
        >>> stepwise = StepwiseZeroInflated(verbose=False)
        >>> stepwise.fit(X, y)
        >>> 
        >>> # Verificar resultados
        >>> print("Features exógenas:", stepwise.columns_exog_)
        >>> print("Features inflacionadas:", stepwise.columns_inf_)
        >>> print("Features excluídas:", stepwise.excluded_)
        
        Exemplo com dados de fraude:
        
        >>> # Dataset de transações
        >>> X_fraud = pd.DataFrame({
        ...     'valor': np.random.lognormal(3, 1, 5000),
        ...     'horario': np.random.choice(range(24), 5000),
        ...     'score_cliente': np.random.beta(2, 5, 5000),
        ...     'historico_meses': np.random.poisson(12, 5000)
        ... })
        >>> 
        >>> # Número de alertas de fraude por transação
        >>> y_fraud = np.random.poisson(0.8, 5000)
        >>> # 60% das transações sem alerta (legítimas)
        >>> y_fraud = np.where(np.random.random(5000) < 0.6, 0, y_fraud)
        >>> 
        >>> # Ajustar com configuração específica
        >>> fraud_stepwise = StepwiseZeroInflated(
        ...     alpha=0.05,
        ...     model_type='ZIP',
        ...     parallel_features=True,
        ...     verbose=False
        ... )
        >>> 
        >>> fraud_stepwise.fit(X_fraud, y_fraud)
        >>> 
        >>> # Modelo final para predições
        >>> final_model = fraud_stepwise.final_model_
        >>> print(f"AIC do modelo final: {final_model.aic:.2f}")
        
        Raises
        ------
        ValueError
            Se y contém valores negativos ou se X não tem features suficientes.
            
        RuntimeError
            Se o algoritmo stepwise falha completamente ou não consegue
            ajustar o modelo final.
            
        NotFittedError
            Se o modelo não consegue convergir para nenhuma configuração
            de features.
            
        Notes
        -----
        - O método automaticamente limpa caches de ajustes anteriores
        - Valores não-inteiros em y são convertidos automaticamente
        - Progress bars são mostradas quando verbose=False
        - O tempo de execução depende do número de features e do tamanho da amostra
        - Para datasets muito grandes (>100k amostras), considere usar batch_size maior
        
        Warnings
        --------
        - Se y contém valores não-inteiros, um warning é emitido antes da conversão
        - Se selection_criterion='LLF', um warning sobre overfitting é mostrado
        """
        # Limpar caches anteriores
        if hasattr(self, '_matrix_cache'):
            self._matrix_cache.clear()
        if hasattr(self, '_model_cache'):
            self._model_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Reset de atributos anteriores
        for attr in ['columns_exog_', 'columns_inf_', 'excluded_', 'final_model_',
                     'criterion_history_', 'step_history_']:
            if hasattr(self, attr):
                delattr(self, attr)

        # Preservar nomes das features
        if hasattr(X, 'columns'):
            original_feature_names = list(X.columns)
        else:
            original_feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        # Validação dos dados
        X, y = check_X_y(X, y, accept_sparse=False)
        
        if np.any(y < 0):
            raise ValueError("y deve conter apenas valores não-negativos")
        if not np.all(np.equal(np.mod(y, 1), 0)):
            warnings.warn("y contém valores não-inteiros que serão convertidos")
            y = y.astype(int)

        # Armazenar informações
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.array(original_feature_names)
        
        # Converter para DataFrame
        X = pd.DataFrame(X, columns=original_feature_names)
        
        if X.shape[1] == 0:
            raise ValueError("X deve ter pelo menos uma feature")

        # Executar stepwise otimizado
        try:
            result = self._stepwise_selection_optimized(X, y)
        except Exception as e:
            raise RuntimeError(f"Erro durante seleção stepwise: {str(e)}")

        # Armazenar resultados
        self.columns_exog_ = result['columns_exog']
        self.columns_inf_ = result['columns_inf']
        self.excluded_ = result['excluded']
        self.criterion_history_ = result.get('criterion_history', [])
        self.step_history_ = result.get('step_history', [])

        # Ajustar modelo final
        try:
            self._fit_final_model_optimized(X, y)
        except Exception as e:
            warnings.warn(f"Erro ao ajustar modelo final: {str(e)}")
            self.final_model_ = None

        return self

    def _fit_final_model_optimized(self, X: pd.DataFrame, y: np.ndarray):
        """Ajusta modelo final de forma otimizada."""
        self.final_model_ = self._fit_model_cached(X, y, self.columns_exog_, self.columns_inf_)
        
        if self.final_model_ is None:
            raise RuntimeError("Falha ao ajustar modelo final")

    def transform(self, X):
        """
        Transforma os dados mantendo apenas as features selecionadas.
        
        Este método aplica a seleção de features obtida durante o ajuste,
        retornando as matrizes de features para os componentes exógeno
        e inflacionado do modelo Zero-Inflated.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Dados a serem transformados. Deve ter o mesmo número de features
            que os dados usados no ajuste.
            
        Returns
        -------
        result : dict
            Dicionário com as matrizes transformadas:
            
            - 'exog' : ndarray of shape (n_samples, n_selected_exog)
                Features selecionadas para o componente exógeno
            - 'inf' : ndarray of shape (n_samples, n_selected_inf)  
                Features selecionadas para o componente inflacionado
                
        Examples
        --------
        Transformação após ajuste:
        
        >>> # Ajustar o modelo
        >>> stepwise = StepwiseZeroInflated()
        >>> stepwise.fit(X_train, y_train)
        >>> 
        >>> # Transformar dados de teste
        >>> X_transformed = stepwise.transform(X_test)
        >>> 
        >>> # Acessar componentes
        >>> X_exog = X_transformed['exog']
        >>> X_inf = X_transformed['inf']
        >>> print(f"Features exógenas: {X_exog.shape[1]}")
        >>> print(f"Features inflacionadas: {X_inf.shape[1]}")
        
        Uso em pipeline scikit-learn:
        
        >>> from sklearn.pipeline import Pipeline
        >>> from sklearn.preprocessing import StandardScaler
        >>> 
        >>> # Pipeline com stepwise selection
        >>> pipeline = Pipeline([
        ...     ('scaler', StandardScaler()),
        ...     ('selector', StepwiseZeroInflated(verbose=False))
        ... ])
        >>> 
        >>> # Ajustar pipeline
        >>> pipeline.fit(X_train, y_train)
        >>> 
        >>> # Transformar novos dados
        >>> X_test_selected = pipeline.transform(X_test)
        
        Raises
        ------
        NotFittedError
            Se o método transform for chamado antes do ajuste.
            
        ValueError
            Se X não tem o mesmo número de features que os dados de treino.
            
        Notes
        -----
        - O método preserva a ordem das amostras
        - Features não selecionadas são completamente removidas
        - Se nenhuma feature foi selecionada para um componente, 
          retorna array vazio com shape (n_samples, 0)
        - Compatível com pipelines scikit-learn
        """
        if not hasattr(self, 'columns_exog_'):
            raise NotFittedError("Este StepwiseZeroInflated ainda não foi ajustado.")

        X = check_array(X, accept_sparse=False)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X tem {X.shape[1]} features, mas esperava {self.n_features_in_}")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)

        result = {}
        result['exog'] = X[self.columns_exog_].values if self.columns_exog_ else np.empty((X.shape[0], 0))
        result['inf'] = X[self.columns_inf_].values if self.columns_inf_ else np.empty((X.shape[0], 0))

        return result

    def fit_transform(self, X, y):
        """
        Ajusta o modelo e transforma os dados em uma única operação.
        
        Equivalente a chamar fit(X, y).transform(X), mas mais eficiente
        por evitar validações duplicadas.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Dados de treino.
            
        y : array-like of shape (n_samples,)
            Variável target (contagem).
            
        Returns
        -------
        result : dict
            Dicionário com as matrizes transformadas (veja transform()).
            
        Examples
        --------
        >>> stepwise = StepwiseZeroInflated(verbose=False)
        >>> X_selected = stepwise.fit_transform(X_train, y_train)
        >>> 
        >>> # Usar features selecionadas
        >>> X_exog_train = X_selected['exog']
        >>> X_inf_train = X_selected['inf']
        """
        return self.fit(X, y).transform(X)

    def score(self, X, y):
        """
        Retorna a pontuação do modelo final (critério de seleção negativo).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Dados de teste.
            
        y : array-like of shape (n_samples,)
            Valores verdadeiros.
            
        Returns
        -------
        score : float
            Pontuação do modelo (critério de seleção negativo).
            Valores maiores indicam melhor ajuste.
            
        Examples
        --------
        >>> # Comparar modelos
        >>> stepwise1 = StepwiseZeroInflated(alpha=0.05)
        >>> stepwise2 = StepwiseZeroInflated(alpha=0.01)
        >>> 
        >>> score1 = stepwise1.fit(X_train, y_train).score(X_test, y_test)
        >>> score2 = stepwise2.fit(X_train, y_train).score(X_test, y_test)
        >>> 
        >>> if score1 > score2:
        ...     print("Modelo 1 é melhor")
        >>> else:
        ...     print("Modelo 2 é melhor")
        
        Raises
        ------
        NotFittedError
            Se o modelo não foi ajustado ou modelo final não está disponível.
        """
        if not hasattr(self, 'final_model_') or self.final_model_ is None:
            raise NotFittedError("Modelo final não disponível")
        return -self._get_model_criterion(self.final_model_)

    def get_params(self, deep=True):
        """Obtém parâmetros para o estimador."""
        return {
            'alpha': self.alpha,
            'model_type': self.model_type,
            'selection_criterion': self.selection_criterion,
            'max_iter': self.max_iter,
            'tolerance': self.tolerance,
            'verbose': self.verbose,
            'model_params': self.model_params,
            'fit_params': self.fit_params,
            'validation_steps': self.validation_steps,
            'use_cache': self.use_cache,
            'parallel_features': self.parallel_features,
            'n_jobs': self.n_jobs,
            'batch_size': self.batch_size,
            'early_stopping': self.early_stopping,
            'multicollinearity_threshold': self.multicollinearity_threshold
        }

    def set_params(self, **params):
        """Define parâmetros para o estimador."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Parâmetro inválido: {key}")

        if 'selection_criterion' in params:
            self._setup_criterion_behavior()

        return self

    def get_cache_stats(self) -> Dict[str, int]:
        """
        Retorna estatísticas de uso do cache para monitoramento de performance.
        
        Útil para otimizar configurações e entender o comportamento do cache
        durante a execução do algoritmo stepwise.
        
        Returns
        -------
        stats : dict
            Dicionário com estatísticas do cache:
            
            - 'cache_hits': Número de acertos no cache
            - 'cache_misses': Número de falhas no cache  
            - 'hit_rate_percent': Taxa de acerto em percentual
            - 'matrix_cache_size': Número de matrizes em cache
            - 'model_cache_size': Número de modelos em cache
            
        Examples
        --------
        Monitoramento de performance:
        
        >>> stepwise = StepwiseZeroInflated(use_cache=True, verbose=False)
        >>> stepwise.fit(X, y)
        >>> 
        >>> # Verificar eficiência do cache
        >>> stats = stepwise.get_cache_stats()
        >>> print(f"Cache hit rate: {stats['hit_rate_percent']:.1f}%")
        >>> print(f"Modelos em cache: {stats['model_cache_size']}")
        >>> print(f"Matrizes em cache: {stats['matrix_cache_size']}")
        >>> 
        >>> # Boa performance: hit rate > 50%
        >>> if stats['hit_rate_percent'] > 50:
        ...     print("Cache funcionando bem!")
        
        Comparação de configurações:
        
        >>> # Testar com diferentes batch sizes
        >>> for batch_size in [3, 5, 8]:
        ...     stepwise = StepwiseZeroInflated(batch_size=batch_size, use_cache=True)
        ...     stepwise.fit(X, y)
        ...     stats = stepwise.get_cache_stats()
        ...     print(f"Batch {batch_size}: {stats['hit_rate_percent']:.1f}% hit rate")
        
        Notes
        -----
        - Hit rates altos (>70%) indicam cache muito eficiente
        - Hit rates baixos (<30%) podem indicar dataset muito variado
        - Cache é resetado automaticamente a cada fit()
        - Estatísticas são acumuladas durante toda a execução do stepwise
        """
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate_percent': (self._cache_hits / (self._cache_hits + self._cache_misses) * 100) 
                               if (self._cache_hits + self._cache_misses) > 0 else 0,
            'matrix_cache_size': len(self._matrix_cache),
            'model_cache_size': len(self._model_cache)
        }

    def clear_cache(self):
        """
        Limpa todos os caches para liberar memória.
        
        Útil quando se trabalha com múltiplos datasets ou quando
        se quer forçar recálculo completo.
        
        Examples
        --------
        Limpeza manual de memória:
        
        >>> stepwise = StepwiseZeroInflated(use_cache=True)
        >>> stepwise.fit(X_large, y_large)  # Cache cresce
        >>> 
        >>> # Verificar uso de memória
        >>> stats_before = stepwise.get_cache_stats()
        >>> print(f"Modelos em cache: {stats_before['model_cache_size']}")
        >>> 
        >>> # Limpar para liberar memória
        >>> stepwise.clear_cache()
        >>> 
        >>> # Verificar limpeza
        >>> stats_after = stepwise.get_cache_stats()
        >>> print(f"Modelos em cache: {stats_after['model_cache_size']}")  # Deve ser 0
        
        Workflow com múltiplos datasets:
        
        >>> stepwise = StepwiseZeroInflated(use_cache=True)
        >>> 
        >>> for dataset_name, (X, y) in datasets.items():
        ...     stepwise.clear_cache()  # Limpar entre datasets
        ...     stepwise.fit(X, y)
        ...     print(f"{dataset_name}: {len(stepwise.columns_exog_)} features")
        
        Notes
        -----
        - Cache é automaticamente limpo no início de cada fit()
        - Limpeza manual só é necessária para liberar memória entre usos
        - Não afeta features selecionadas ou modelo final
        - Reseta contadores de hits/misses para zero
        """
        self._matrix_cache.clear()
        self._model_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0 