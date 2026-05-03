import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
from pymongo import MongoClient
from tqdm import tqdm

# ========== CONFIGURATION ==========
MIN_FRIENDS = 10  # Only include users with MORE than 20 friends
BATCH_SIZE = 5000
# ================================

client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=30000)
db = client["yelp"]
users_collection = db["user"]

print("Available databases:", client.list_database_names())


print("=== Configuration ===")
print(f"Minimum friends required: > {MIN_FRIENDS}")
print(f"Total users in DB: {users_collection.count_documents({})}")

# ========== STEP 1: Pre-filter Users in MongoDB ==========
print("\n=== Filtering Users ===")

# Count users meeting criteria (estimate by sampling first)
sample = users_collection.find_one()
if sample and isinstance(sample.get("friends"), str):
    friends_count = len(sample["friends"].split(","))
    print(f"Sample user friends count: {friends_count}")

# Count users with > MIN Friends (using aggregation to count commas)
# MongoDB aggregation to count enemies (split and count)
filter_pipeline = [
    {
        "$addFields": {
            "friendsArray": {"$split": ["$friends", ", "]},
            "friendCount": {"$size": {"$split": ["$friends", ", "]}},
        }
    },
    # Alternative: Count commas + 1 (faster for large datasets)
    {
        "$addFields": {
            "friendCount": {
                "$add": [
                    {"$size": {"$split": ["$friends", ","]}},
                    0,  # placeholder
                ]
            }
        }
    },
    {"$match": {"friendCount": {"$gt": MIN_FRIENDS}}},
    {"$count": "matching_users"},
]

# Since $size on split string might timeout, let's do it in Python with pre-filter
print("Fetching users and filtering in Python (faster for this case)...")

edges = []
user_count = 0
filtered_count = 0

cursor = users_collection.find({}, {"user_id": 1, "friends": 1, "name": 1, "_id": 0})
cursor.batch_size(BATCH_SIZE)

for user in tqdm(cursor, desc="Processing users"):
    user_count += 1
    user_id = user.get("user_id")
    friends_str = user.get("friends", "")
    name = user.get("name", "Unknown")

    if friends_str and isinstance(friends_str, str):
        # Split and count
        friends_list = [f.strip() for f in friends_str.split(",") if f.strip()]
        friend_count = len(friends_list)

        # Filter: only include users with > MIN_FRIENDS
        if friend_count > MIN_FRIENDS:
            filtered_count += 1

            # Create edges
            for friend_id in friends_list:
                if friend_id:
                    edges.append(
                        {
                            "source": user_id,
                            "target": friend_id,
                            "user_name": name,
                            "friend_count": friend_count,
                        }
                    )

    if user_count % 200000 == 0:
        print(
            f"  Processed {user_count} users, {filtered_count} filtered, {len(edges)} edges..."
        )

print(f"\n=== Filtering Results ===")
print(f"Total users processed: {user_count}")
print(f"Users with > {MIN_FRIENDS} friends: {filtered_count}")
print(f"Total edges created: {len(edges)}")
print(f"Reduction: {100 * (1 - filtered_count / user_count):.1f}% fewer users")

if filtered_count == 0:
    print(f"\n⚠️ No users have more than {MIN_FRIENDS} friends!")
    print("Try lowering MIN_FRIENDS to 5 or 10.")
    exit()

# ========== STEP 2: Clean Edges (Remove Duplicate Friend IDs) ==========
print("\nCleaning edges...")
df_edges = pd.DataFrame(edges)

# Remove edges where friend is not in our filtered set
# First, get all unique user_ids that are in our filtered set
filtered_user_ids = set(df_edges["source"].unique())

# Filter edges to only include friends that are also in our filtered set
df_edges = df_edges[df_edges["target"].isin(filtered_user_ids)]

print(f"Edges after filtering to internal connections: {len(df_edges)}")

# Remove duplicates (A→B and B→A are same undirected edge)
df_edges_unique = df_edges[["source", "target"]].copy()
df_edges_unique["sorted_edge"] = df_edges_unique.apply(
    lambda row: tuple(sorted([row["source"], row["target"]])), axis=1
)
df_edges_unique = df_edges_unique.drop_duplicates(subset="sorted_edge")
df_edges_clean = df_edges_unique[["source", "target"]]

print(f"Unique undirected edges: {len(df_edges_clean)}")

# ========== STEP 3: Build Graph ==========
print("\nBuilding NetworkX graph...")
G = nx.Graph()
G.add_edges_from(zip(df_edges_clean["source"], df_edges_clean["target"]))

print(f"Network nodes: {G.number_of_nodes()}")
print(f"Network edges: {G.number_of_edges()}")

if G.number_of_nodes() == 0:
    print("ERROR: No nodes in graph!")
    exit()

# ========== STEP 4: Statistics ==========
print("\n=== Network Statistics ===")
print(f"Average degree: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
print(f"Density: {nx.density(G):.6f}")
print(f"Connected components: {nx.number_connected_components(G)}")
print(f"Average clustering: {nx.average_clustering(G):.4f}")

largest_cc = max(nx.connected_components(G), key=len)
print(
    f"Largest CC: {len(largest_cc)} nodes ({100 * len(largest_cc) / G.number_of_nodes():.1f}%)"
)

# ========== STEP 5: Static Plot ==========
print("\nGenerating static plot...")
G_lcc = G.subgraph(largest_cc)

sample_size_static = min(400, G_lcc.number_of_nodes())
sample_nodes_static = list(nx.random_sample(G_lcc.nodes(), sample_size_static))
G_sample_static = G_lcc.subgraph(sample_nodes_static)

plt.figure(figsize=(14, 10))
pos = nx.spring_layout(G_sample_static, k=0.4, iterations=40, seed=42)

node_sizes = [G_sample_static.degree(n) * 5 for n in G_sample_static.nodes()]

nx.draw(
    G_sample_static,
    pos,
    node_size=node_sizes,
    node_color="#FF6B6B",
    edge_color="#888",
    with_labels=False,
    alpha=0.7,
    width=0.4,
)

plt.title(
    f"Yelp User Network (Users with > {MIN_FRIENDS} friends)\nSample: {sample_size_static} from Largest CC",
    fontsize=14,
)
plt.axis("off")
plt.tight_layout()
plt.savefig("user_network_filtered_static.png", dpi=150)
print("Saved: user_network_filtered_static.png")
plt.show()

# ========== STEP 6: Interactive Plot ==========
print("\nGenerating interactive plot...")
sample_size_interactive = min(250, G_sample_static.number_of_nodes())
sample_nodes_interactive = list(
    nx.random_sample(G_sample_static.nodes(), sample_size_interactive)
)
G_sample = G_sample_static.subgraph(sample_nodes_interactive)

pos_sample = nx.spring_layout(G_sample, k=0.4, iterations=40, seed=42)

edge_x, edge_y = [], []
for edge in G_sample.edges():
    x0, y0 = pos_sample[edge[0]]
    x1, y1 = pos_sample[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

edge_trace = go.Scatter(
    x=edge_x,
    y=edge_y,
    line=dict(width=0.5, color="#888"),
    hoverinfo="none",
    mode="lines",
)

node_x = [pos_sample[node][0] for node in G_sample.nodes()]
node_y = [pos_sample[node][1] for node in G_sample.nodes()]

node_degrees = dict(G_sample.degree())
node_sizes_plotly = [d * 8 for d in node_degrees.values()]

node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    mode="markers",
    hoverinfo="text",
    marker=dict(
        showscale=True,
        colorscale="Reds",
        size=node_sizes_plotly,
        color=list(node_degrees.values()),
        colorbar=dict(title="Degree", thickness=12, x=1.05),
        line=dict(width=1, color="white"),
    ),
)
node_trace.text = [f"User: {n[:25]}...<br>Degree: {d}" for n, d in node_degrees.items()]

fig = go.Figure(
    data=[edge_trace, node_trace],
    layout=go.Layout(
        title=f"Filtered Yelp Network<br>Only Users with > {MIN_FRIENDS} Friends (Sample: {sample_size_interactive})",
        titlefont=dict(size=18),
        showlegend=False,
        hovermode="closest",
        margin=dict(b=10, l=10, r=10, t=60),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    ),
)

fig.write_html("user_network_filtered_interactive.html")
print("Saved: user_network_filtered_interactive.html")
fig.show()

# ========== STEP 7: Export ==========
df_edges_clean.to_csv("user_network_filtered_edges.csv", index=False)
print(f"\nSaved: user_network_filtered_edges.csv ({len(df_edges_clean)} edges)")

# ========== STEP 8: Top 15 Users ==========
print(f"\n=== Top 15 Users with > {MIN_FRIENDS} Friends ===")
print(f"{'Rank':<6} {'User ID':<28} {'Degree':>10} {'Name'}")
print("-" * 70)

degrees = dict(G.degree())
# Get friend count from original df_edges
user_friend_counts = df_edges.groupby("source")["friend_count"].first()
top_users = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:15]

for rank, (user_id, degree) in enumerate(top_users, 1):
    friend_count = user_friend_counts.get(user_id, "N/A")
    user_display = user_id[:26] if len(user_id) > 26 else user_id
    print(f"{rank:<6} {user_display:<28} {degree:>10} {friend_count}")
