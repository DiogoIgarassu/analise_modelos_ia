import smtplib
import cores as cor

def enviar_email(assunto, mensagem):
    remetente = 'diogoigarassu@gmail.com'
    senha = 'dymgeqcnmuanbumt'
    destinatario = 'euamofisica2006@hotmail.com, diogoigarassu@gmail.com'

    corpo = f"Subject: {assunto}\n\n{mensagem}"
    corpo = corpo.encode('utf-8')

    with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.login(remetente, senha)
        smtp.sendmail(remetente, destinatario, corpo)

        print(cor.FIM, cor.BRA, cor.BPRE, "\nE-mail enviado para informar o término do processo..\n", cor.FIM)

#enviar_email("teste de email ML", "Teste para informar fins dos cálculos")