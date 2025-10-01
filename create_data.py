import numpy as np
df = pd.read_csv("/home/alex/Network/UMDropBox/JensenLab/data/Rapid KO/SMU_UA159/QC Analysis/growth_database.csv")

intergenic_idx = np.array(df['New_ID'][new_id.str.contains('intergenic')].index)
colony_growth = np.array(df['colony_growth'].iloc[intergenic_idx])
liquid_growth = np.array(df['liquid_growth'].iloc[intergenic_idx])

essential_intergenic = [ jdx for jdx, i, j in zip(range(len(intergenic_idx)), colony_growth, liquid_growth) if (i == j and i == 'N') or (i != j)]
df.iloc[intergenic_idx[essential_intergenic]].to_csv("/home/alex/Documents/Mutans Optimization/data/essential_intergenic.csv")

     
entergenic_idx = np.array(df['New_ID'][new_id.str.contains('SMU_')].index)
colony_growth = np.array(df['colony_growth'].iloc[entergenic_idx])
liquid_growth = np.array(df['liquid_growth'].iloc[entergenic_idx])

essential_genes = [ jdx for jdx, i, j in zip(range(len(entergenic_idx)), colony_growth, liquid_growth) if (i == j and i == 'N') or (i != j)]
df.iloc[entergenic_idx[essential_genes]].to_csv("/home/alex/Documents/Mutans Optimization/data/essential_entergenic.csv")

idx = np.array(df.index)
colony_growth = np.array(df['colony_growth'].iloc[idx])
liquid_growth = np.array(df['liquid_growth'].iloc[idx])

essential = [ jdx for jdx, i, j in zip(range(len(idx)), colony_growth, liquid_growth) if (i == j and i == 'N') or (i != j)]
df.iloc[idx[essential]].to_csv("/home/alex/Documents/Mutans Optimization/data/full_essential.csv")
