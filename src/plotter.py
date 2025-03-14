import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':

    csv_files = glob.glob("data-main/*.csv")  # Modifica il path se necessario

    # Creiamo una lista per raccogliere i dati
    data_list = []

    # Leggiamo tutti i file CSV
    for file in csv_files:
        if "False" in file:
            model_type = "dense"
        elif "True-sparsity-0.3" in file:
            print(file)
            model_type = "Sparse (0.3)"
        elif "True-sparsity-0.5" in file:
            print(file)
            model_type = "Sparse (0.5)"
        elif "True-sparsity-0.7" in file:
            print(file)
            model_type = "Sparse (0.7)"
        elif "True-sparsity-0.9" in file:
            print(file)
            model_type = "Sparse (0.9)"
        else:
            continue  # Ignora file non pertinenti

        df = pd.read_csv(file)
        df["Model"] = model_type
        data_list.append(df)

    # Concatenazione di tutti i dati in un unico DataFrame
    data = pd.concat(data_list, ignore_index=True)

    print(data['Model'].unique())

    # Creazione dei boxplot
    plt.figure(figsize=(12, 5))

    # Boxplot per inference time
    plt.subplot(1, 2, 1)
    sns.boxplot(x="Model", y="Time", data=data, hue='Model', palette="Set2")
    plt.title("Inference Time Comparison")
    plt.ylabel("Time (s)")

    # Boxplot per accuracy
    plt.subplot(1, 2, 2)
    sns.boxplot(x="Model", y="Accuracy", data=data,  hue='Model', palette="Set2")
    plt.title("Accuracy Comparison")
    plt.ylabel("Accuracy")

    plt.tight_layout()
    plt.savefig('comparison.pdf', dpi=500)