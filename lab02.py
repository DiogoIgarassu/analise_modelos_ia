def escreva_relatorio(rodada, space, param_grid, detalhes_info, formatted_time):
    with open(f"relatorios/relatorio_{formatted_time}.txt", "w") as arquivo:
        arquivo.write(f"========================================================\n"
                      f"                Relatório de Teste de Parâmetros\n"
                      f"========================================================\n\n"
                      f"Número de rodada de teste: {rodada}\n"
                      f"========================================================\n"
                      f"                     Parâmetros Iniciais\n"
                      f"========================================================\n\n"
                      f"Parametros para árvore de decisão:\n"
                      f"criterion: {space['criterion']}\n"
                      f"min_samples_split: {space['min_samples_split']}\n"
                      f"max_depth: {space['max_depth']}\n"
                      f"min_samples_leaf: {space['min_samples_leaf']}\n"
                      f"Parametros para SVM:\n"
                      f"param_grid C: {param_grid['C']}\n"
                      f"param_grid gamma: {param_grid['gamma']}\n"
                      f"param_grid kernel: {param_grid['kernel']}\n"
                      f"========================================================\n\n")
        for detalhe in detalhes_info:
            arquivo.write(detalhe)
