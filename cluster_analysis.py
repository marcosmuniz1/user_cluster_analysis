import numpy as np
import pandas as pd
from kmodes.kmodes import KModes
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("RetailHabits.csv")
codes=pd.read_csv("Codebook.csv")

# GOT A DATASET WITH CONSUMER DATA ON WHAT RETAIL ITEMS THEY BOUGHT AND FREQUENCY OF ENCAGEMENT IN DIFFERENT RETAIL CHANNELS
#CAMBIO PIO 3
#CAMBIO PIO nuevo

# GOAL IS TO IDENTIFY CLUSTERS OF CONSUMERS AND HOW THEY ARE LIKELY TO ENGAGE IN COMMERCE, AND WETHER CERTAIN DEMO GROUPS
## ARE MORE OR LESS LIKELY TO BELONG TO A GIVEN CLUSTER


print("dataset contains a couple of badly formatted binary columns then frequency indications for certain shopping behaviours\n")
for c in df.columns:
    print(f"{c}:")
    print(df[c].unique())
    print("***")
    
print("i will start with the binary stuff given that i can treat them as a group and work on them parallely")

items=list(['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10'])

# extracting purchase type from columns

item_names=list()
for a in items:
    for u in df[a].unique():
        if u == '0':
            continue
        else:
            item_names.append(u)

# renaming columns so they are more informative

counter=0
for i in range(len(item_names)):
    df=df.rename(columns={items[i]:item_names[i]})
df.head()

# binarize columns: true=shopped for item in last 30 days

for i in item_names:
    df[i]=df[i]!='0'
    

# i'd like to compare share of shoppers per category in a single chart

shoppers=list()
subset_size=len(df)

for i in item_names:
    shoppers.append(len(df[df[i]==True]))

shoppers=np.array(shoppers)
shoppers=shoppers/subset_size
shoppers=pd.Series(dict(zip(item_names,shoppers)))

shoppers.sort_values().plot(kind='bar')
plt.show()

print("A1 to A4 seem to be nonrecurrent and therefore uninformative for recurrent purchase behaviour so i'll take them out")

item_names=item_names[4:]


# what about shopper overlappings

overlaps = pd.DataFrame(0, columns=item_names, index=item_names, dtype=float)

# Iterate through the item names
for item1 in item_names:
    for item2 in item_names:
        if item1 == item2:
            overlaps.at[item1, item2] =1
        if item1 != item2:
            shoppers_item1 = df[df[item1] == True].shape[0]
            shoppers_both = df[(df[item1] == True) & (df[item2] == True)].shape[0]
            if shoppers_item1 != 0:
                overlaps.at[item1, item2] = shoppers_both / shoppers_item1
                
plt.figure(figsize=(8, 6))
sns.heatmap(overlaps, annot=True, cmap='YlGnBu', vmin=0, vmax=1)
plt.show()


# Custering #################

df_binary = df[item_names]
n_clusters=[2,3,4]
results_df = pd.DataFrame(columns=
                          ['Number of Clusters', 'Silhouette Score', 'Davies-Bouldin Index', 'Calinski-Harabasz Index'])

for n in n_clusters:
    km = KModes(n_clusters=n, init='Huang', n_init=5, verbose=0)
    clusters = km.fit_predict(df_binary)
    
    silhouette_avg = silhouette_score(df_binary, clusters)
    db_score = davies_bouldin_score(df_binary, clusters)
    ch_score = calinski_harabasz_score(df_binary, clusters)
    
    # Append results to the DataFrame
    results_df = results_df.append({
        'Number of Clusters': n,
        'Silhouette Score': silhouette_avg,
        'Davies-Bouldin Index': db_score,
        'Calinski-Harabasz Index': ch_score
    }, ignore_index=True)
