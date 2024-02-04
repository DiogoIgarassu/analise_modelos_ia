import smtplib
import cores as cor

def obter_credenciais():
    with open('config.txt', 'r') as file:
        lines = file.readlines()
        credenciais = {}
        for line in lines:
            key, value = line.strip().split('=')
            credenciais[key] = value
        return credenciais

def enviar_email(assunto, mensagem):
    credenciais = obter_credenciais()
    remetente = credenciais['email']
    senha = credenciais['senha']
    destinatario = credenciais['destinatario']

    corpo = f"Subject: {assunto}\n\n{mensagem}"
    corpo = corpo.encode('utf-8')

    with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.login(remetente, senha)
        smtp.sendmail(remetente, destinatario, corpo)

        print(cor.FIM, cor.BRA, cor.BPRE, "\nE-mail enviado para informar o término do processo..\n", cor.FIM)

enviar_email("teste de email ML", "Teste para informar fins dos cálculos")
