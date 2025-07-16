import os

pasta = r"C:\Users\patrick.rosa\Desktop\iaedit\data\editais"  # Caminho absoluto para o diretório 'pdfs'

for arquivo in os.listdir(pasta):
    print(arquivo)