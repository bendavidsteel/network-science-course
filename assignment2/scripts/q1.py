import os
import pandas as pd
import networkx as nx

def load_graph():
    data = pd.read_csv('/home/nicola-linux/Documents/MilaUdeM/COMP599/assignment2/data/email-Enron/email-Enron.txt', sep=" ", header=None)
    data.columns = ["sender", "recipient", "timestamp"]

    # aggregate counts of sender and recipient emails
    df = data.groupby(['sender','recipient']).size().reset_index(name='weight')
    
    # convert to graph
    G=nx.from_pandas_edgelist(df, 'sender', 'recipient', 'weight')

    return G

def main():
    G = load_graph()
    
    # degree centrality
    DC = sorted(G.degree(weight='weight'), key=lambda x: x[1], reverse=True)
    print(DC[:5])

    # eigenvector centrality
    EC = nx.eigenvector_centrality(G, weight='weight')
    EC = dict(sorted(EC.items(), key=lambda item: item[1], reverse=True))
    print(list(EC.items())[:5])

    # katz centrality
    #KC = nx.katz_centrality(G, tol=2.0e-6, weight='weight')
    #KC = dict(sorted(KC.items(), key=lambda item: item[1], reverse=True))
    #print(list(KC.items())[:5])

    # closeness centrality
    CC = nx.closeness_centrality(G)
    CC = dict(sorted(CC.items(), key=lambda item: item[1], reverse=True))
    print(list(CC.items())[:5])

if __name__ == "__main__":
    main()