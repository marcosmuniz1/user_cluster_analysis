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
    km = KModes(n_clusters=n, init='Huang', n_init=10, verbose=0)
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


km = KModes(n_clusters=4, init='Huang', n_init=10, verbose=0)
clusters = km.fit_predict(df_binary)
df['cluster']=clusters

print(f"cluster sizes for {n_clusters} splits")
print(df['cluster'].value_counts())

shoppers = df.groupby(by='cluster')[item_names].mean()
shoppers.plot(kind='bar', stacked=False)
plt.title("3) Share of purchasers by cluster - 4 splits")
plt.savefig('purchasers_by_cluster4.png')

print("notice: saved chart with how many people in each cluster are buying each item")
print("***\n")

print("looks like there is a progression in the number of purchase types by cluster, with the note that a relevant chunk buys few stuff but still fuel ")
print("this could be interesting if i had to analyze strategies to reach underserved consumers")
print("apart from that, i'd think a scale of how many item categories they spent in through the last month would be clearer to communicate\n***\n")
#print("the three cluster model that nonbasic purchases such as clothing or alcohol define belonging to a subset of consumers. clusters 0 and 1 are somewhat more proximate, with gasoline being a key differentiator")

print("let's see if there are implicit intertwines in channel usage\n")
print("step one is to see how frequent is each channel\n")

channel_u_columns = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
channel_mapping= {
    'B1':"In-store",
    'B2':"Online with delivery",
    'B3':"Curbside",
    'B4':"Pickup",
    'B5':"Subscription",
    'B6':"Same-day app",
    'B7':"Marketplace"
}

df.rename(columns=channel_mapping, inplace=True)
channel_u_columns=df.columns[-11:-4].values

# Define the factor ordering and convert columns to ordered categorical data
factor_ordering = [
    "I have not done this in the last 30 days.",
    "Not weekly but monthly",
    "Not daily but weekly",
    "Daily"
]

for column in channel_u_columns:
    df[column] = pd.Categorical(df[column], categories=factor_ordering, ordered=True)
    
n_levels = len(factor_ordering)
color_palette = sns.color_palette("YlGnBu", n_levels)

# Create a figure with subplots for each column
fig, axes = plt.subplots(1, len(channel_u_columns), figsize=(15, 5),sharey=True)

# Iterate through columns and plot the relative frequencies with color gradients
for i, column in enumerate(channel_u_columns):
    # Calculate relative frequencies
    relative_freq = df[column].value_counts(normalize=True).loc[factor_ordering]
    
    # Plot a bar chart with color gradients
    sns.barplot(x=relative_freq.index, y=relative_freq.values, ax=axes[i], palette=color_palette)
    axes[i].set_title(column)
    axes[i].set_ylabel("Relative Frequency")
    axes[i].set_xticklabels([])

plt.suptitle("Usage Frequency, by Channel")
plt.tight_layout()
plt.savefig('channel_usage_freqs.png')
print("note: saved a chart with univariate frequency distributions. most channels are rare except in-store and online with delivery ")
print("this suggests two or three groups may be enough to differentiate the consumer base, but let's see ")
print("on to building the clusters")

#there are potentially three possible mappings for these factors, let's start with number 3 and then we can test the others

# basic encoding with not done=0
factor_mapping1 = {
    "I have not done this in the last 30 days.":0,
    "Not weekly but monthly":1,
    "Not daily but weekly":2,
    "Daily":3
}

# # basic encoding as if we had no domain knowledge
factor_mapping2 = {
    "I have not done this in the last 30 days.":1,
    "Not weekly but monthly":2,
    "Not daily but weekly":3,
    "Daily":4
}

#based on "days a month"
factor_mapping3= {
    "I have not done this in the last 30 days.":0,
    "Not weekly but monthly":1,
    "Not daily but weekly":4,
    "Daily":30
}

print("aside from the mapping, ill swap nans for zeros (if they do not answer the question i'ts because they purchased nothing)")
mapped_df = df[channel_u_columns].applymap(factor_mapping3.get).fillna(0)

print(" based on the first glance at the data i'll try a 3 cluster model\n")

km = KModes(n_clusters=3, init='Huang', n_init=10, verbose=0)
clusters = km.fit_predict(mapped_df)
mapped_df['cluster']=clusters

print(f"cluster sizes for 3 splits")
print(mapped_df['cluster'].value_counts())

#revert the mapping so its clearer for visualization
labels_df=df[channel_u_columns].copy().fillna("I have not done this in the last 30 days.")
labels_df['cluster']=mapped_df['cluster'].copy()

# Get unique clusters
unique_clusters = labels_df['cluster'].unique()

# Create a figure with subplots for each cluster and each column
fig, axes = plt.subplots(len(unique_clusters), len(labels_df.columns) - 1, figsize=(15, 5))

# Iterate through clusters
for i, cluster in enumerate(unique_clusters):
    cluster_df = labels_df[labels_df['cluster'] == cluster]
    
    # Iterate through columns and plot the relative frequencies with color gradients for each cluster
    for j, column in enumerate(cluster_df.columns[:-1]):  # Exclude the 'cluster' column
        ax = axes[i, j]
        relative_freq = cluster_df[column].value_counts(normalize=True).loc[factor_ordering]
        sns.barplot(x=relative_freq.index, y=relative_freq.values, ax=ax, palette=color_palette)
        ax.set_title(f"Cluster {cluster}, {column}")
        ax.set_xticklabels([])

# Add a common y-axis label
for ax in axes[:, 0]:
    ax.set_ylabel("Relative Frequency")

# Add a chart title
plt.suptitle("Factor Frequencies by Column per Cluster")

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

print("intuition behind clusters is pretty straight forward: mainly in-store shoppers, those who have incorporated online ordering, and those who use multiple channels")

print(" can't jurt to see how it looks with 4 clusters\n")

km = KModes(n_clusters=4, init='Huang', n_init=10, verbose=0)
clusters = km.fit_predict(mapped_df)
mapped_df['cluster']=clusters

print(f"cluster sizes for 4 splits")
print(mapped_df['cluster'].value_counts())

#revert the mapping so its clearer for visualization
labels_df=df[channel_u_columns].copy().fillna("I have not done this in the last 30 days.")
labels_df['cluster']=mapped_df['cluster'].copy()

# Get unique clusters
unique_clusters = labels_df['cluster'].unique()

# Create a figure with subplots for each cluster and each column
fig, axes = plt.subplots(len(unique_clusters), len(labels_df.columns) - 1, figsize=(15, 5))

# Iterate through clusters
for i, cluster in enumerate(unique_clusters):
    cluster_df = labels_df[labels_df['cluster'] == cluster]
    
    # Iterate through columns and plot the relative frequencies with color gradients for each cluster
    for j, column in enumerate(cluster_df.columns[:-1]):  # Exclude the 'cluster' column
        ax = axes[i, j]
        relative_freq = cluster_df[column].value_counts(normalize=True).loc[factor_ordering]
        sns.barplot(x=relative_freq.index, y=relative_freq.values, ax=ax, palette=color_palette)
        ax.set_title(f"Cluster {cluster}, {column}")
        ax.set_xticklabels([])

# Add a common y-axis label
for ax in axes[:, 0]:
    ax.set_ylabel("Relative Frequency")

# Add a chart title
plt.suptitle("Factor Frequencies by Column per Cluster")

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

#pretty much same intuition, only that new cluster uses the channels with more frequency