from django.test import TestCase

# Create your tests here.
# def send_email(message, from_email):
#     smtp_port = EMAIL_PORT
#     smtp_server = EMAIL_HOST
#     from_email = EMAIL_HOST_USER
#     to_email = 'frederic.ferreira66@yahoo.com'

#     passwd = EMAIL_HOST_PASSWORD
#     message = 'Nouveau message test'
#     simple_email_context = ssl.create_default_context()
#     try:
#         print('connexion au serveur...')
#         TIE_server = smtplib.SMTP(smtp_server, smtp_port)
#         TIE_server.starttls(context=simple_email_context)
#         TIE_server.login(from_email, passwd)
#         print('Connexion au serveur réussie')
#         print()
#         print(f"envoi un email à {to_email}")
#         TIE_server.sendmail(from_email, to_email, message)
#         print(f'email envoyé avec succès à {to_email}')
#     except Exception as e:
#         print(e)

