import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Εισαγωγή των δεδομένων
data = pd.read_csv("C:/Users/John/Documents/Google Drive/Pattern-Recognision/country-data.csv")


# Προεπεξεργασία των δεδομένων
numerical_data = data.drop("Country", axis=1)
# Αρχικά ξεχωρίζουμε τα δεδομένα την στήλη με την χώρα
# Αφαιρούμε την μέση τιμή και διαιρούμε με την τυπική απόκλιση
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler = scaler.fit(numerical_data)
transformed = pd.DataFrame(scaler.transform(numerical_data), columns=["Population","Area","PopulationDensity","Coastline","Migration","InfantMortality","GDP","Literacy","Phones","Birthrate","Deathrate"])

# Κάνουμε χρήση του αλγορίθμου PCA για να μειώσουμε τις διαστάσεις των δεδομένων
from sklearn.decomposition import PCA
pca = PCA()
pca = pca.fit(transformed)
pca_transformed = pca.transform(transformed)
eigenvalues = pca.explained_variance_
eigenvectors = pca.components_

# Με το παρακάτω διάγραμμα παρατηρούμε ότι η πληροφορία βρίσκεται στις 5
# πρώτες κύριες συνιστώσες
plt.bar(range(len(eigenvalues)), eigenvalues/sum(eigenvalues))
plt.show()

# Εφαρμόζουμε τον αλγόριθμο PCA και κρατάμε τις 5 πρώτες κύριες συνιστώσες
pca = PCA(n_components=5)
pca = pca.fit(transformed)
pca_transformed = pd.DataFrame(pca.transform(transformed))

pca_inverse = pd.DataFrame(pca.inverse_transform(pca_transformed), columns=numerical_data.columns)

# Επιβεβαιώνουμε ότι κρατήσαμε το μεγαλύτερο μέρος της πληροφόριας
# Η απώλεια πληροφορίας είναι ίδη με info_loss = 0.178354752218584 ή περίπου 18%
info_loss = 1 - (eigenvalues[0] + eigenvalues[1] + eigenvalues[2] + eigenvalues[3] + eigenvalues[4])/sum(eigenvalues)  
print(info_loss)

#Μοντελοποίηση
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import silhouette_score

slc = []
for i in range(2, 21):
    clustering = AgglomerativeClustering(n_clusters=i, linkage="complete").fit(pca_inverse)
    slc.append(silhouette_score(pca_inverse, clustering.labels_))

# Παρατηρούμε ότι όσο αυξάνονται οι ομάδες η τιμή της μετρικής silhouette μικραίνει
# Συνεπώς είναι βέλτιστο να ομαδοποιήσουμε τις χώρες το πολύ σε τέσσερις ομάδες
plt.plot(range(2, 21), slc)
plt.xticks(range(2, 21), range(2, 21))
plt.show()


clustering = AgglomerativeClustering(n_clusters=4, linkage="complete").fit(pca_inverse)
