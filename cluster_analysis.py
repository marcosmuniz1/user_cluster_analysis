import numpy as np
import pandas as pd
from kmodes.kmodes import KModes
from sklearn.preprocessing import MinMaxScaler
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


print("dataset contains a couple of badly formatted binary columns, then frequency indications for certain shopping behaviours\n")
for c in df.columns:
    print(f"{c}:")
    print(df[c].unique())
    print("***")
    
print("i will start with the binary stuff given that i can treat them as a group and work on them parallely")
print("***\n")

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

shoppers = df[item_names].mean()
shoppers.sort_values().plot(kind='bar')
plt.title("1) Share of people who made purchase of select products in last 30 days")
plt.figure(figsize=(8, 6))
plt.savefig('share_purchasers_overall.png') 

print("notice: saved a chart with share of purchasers per category overall")
print("A1 to A4 seem to be nonrecurrent and therefore uninformative for recurrent purchase behaviour so i took them out")
print("***\n")

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
plt.title("2) Share of column purchasers who also purchased row")
plt.savefig('purchase_interactions.png')
print("notice: saved a heatmap with item-to-item purchase interactions")
print("***\n")

# Custering #################

print("clusters may help us have a better grasp on how purchase patterns between categories are intertwined")
print("used KModes, will test different n_cluster configurations but init and n_init will remain fixed at Huang and =5")

df_binary = df[item_names]
#let's test different cluster n choices via goodnes of cluster metrics
n_clusters=[2,3,4]
# empty df for results
results_df = pd.DataFrame(columns=
                          ['Number of Clusters', 'Silhouette Score', 'Davies-Bouldin Index', 'Calinski-Harabasz Index'])

# loop through configurations and calculate appropiate metrics
for n in n_clusters:
    km = KModes(n_clusters=n, init='Huang', n_init=5, verbose=0)
    clusters = km.fit_predict(df_binary)
    
    silhouette_avg = silhouette_score(df_binary, clusters) # how similar an object is to its own cluster (cohesion) compared to other clusters 
    db_score = davies_bouldin_score(df_binary, clusters) # average similarity between each cluster and its most similar cluster
    ch_score = calinski_harabasz_score(df_binary, clusters) #ratio of the between-cluster variance to within-cluster variance
    
    # Append results to the DataFrame
    results_df = results_df.append({
        'Number of Clusters': n,
        'Silhouette Score': silhouette_avg,
        'Davies-Bouldin Index': db_score,
        'Calinski-Harabasz Index': ch_score
    }, ignore_index=True)
    
    
# I need to normalize to make feasible comparison between the metrics, ill use minmax for high=good and reverse the
# minmax when the opposite is true")

results_df_altered=results_df.copy()

# high=good
scaler = MinMaxScaler()
columns_to_scale = ['Silhouette Score', 'Calinski-Harabasz Index']
scaled_columns = scaler.fit_transform(results_df_altered[columns_to_scale])
results_df_altered[columns_to_scale] = scaled_columns

# high=bad
Max_dbi = results_df_altered['Davies-Bouldin Index'].max()
Min_dbi = results_df_altered['Davies-Bouldin Index'].min()

results_df_altered['Davies-Bouldin Index'] = 1 - (results_df_altered['Davies-Bouldin Index'] - Min_dbi) / (Max_dbi - Min_dbi)


results_df_altered['Overall']=(results_df_altered['Silhouette Score']+
                                       results_df_altered['Calinski-Harabasz Index']+
                                       results_df_altered['Davies-Bouldin Index'])/3

# create an overall measure to make a decision

n_clusters=results_df_altered[results_df_altered['Overall']==results_df_altered['Overall'].max()]['Number of Clusters'].mean()
print(f"{n_clusters} is the preferred cluster size based on cluster optimization\n")

print(" i saw the results for 4 clusters and it wasnt very insighful, plus 4 categories is a bit too many, so i reduced to 3\n")
km = KModes(n_clusters=3, init='Huang', n_init=10, verbose=0)
clusters = km.fit_predict(df_binary)
df['cluster']=clusters

print("cluster sizes")
print(df['cluster'].value_counts())

shoppers = df.groupby(by='cluster')[item_names].mean()
shoppers.plot(kind='bar', stacked=False)
plt.title("3) Share of purchasers by cluster")
plt.savefig('purchasers_by_cluster.png')

print("notice: saved chart with how many people in each cluster are buying each item")
print("***\n")

print("the three cluster model that nonbasic purchases such as clothing or alcohol define belonging to a subset of consumers. clusters 0 and 1 are somewhat more proximate, with gasoline being a key differentiator")