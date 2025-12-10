import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn import tree

# Se define la ruta para el directorio de datos sin procesar
path_data = Path.cwd().parent / 'data' / 'raw'

# Se carga los archivos csv
promotions_raw = pd.read_csv(path_data / "promotions.csv")
customers_raw = pd.read_csv(path_data / "customers.csv")
customer_labels_raw = pd.read_csv(path_data / "customer_labels.csv")
products_raw = pd.read_csv(path_data / "products.csv")
stores_raw = pd.read_csv(path_data / "stores.csv")
inventory_raw = pd.read_csv(path_data / "inventory.csv")
transactions_raw = pd.read_csv(path_data / "transactions.csv")

# Se genera copia de los dfs cargados originalmente
promotions = promotions_raw.copy()
customers = customers_raw.copy()
customer_labels = customer_labels_raw.copy()
products = products_raw.copy()
stores = stores_raw.copy()
inventory = inventory_raw.copy()
transactions = transactions_raw.copy()

def es_promo(sku, fecha, pmt):
    fecha = pd.to_datetime(fecha)
    promotions_temp = pmt.loc[pmt["sku"] == sku, ["fecha_inicio", "fecha_fin", "descuento_pct"]]
    descuento_pct = promotions_temp.loc[(
        (promotions_temp["fecha_inicio"] <= fecha) &
        (promotions_temp["fecha_fin"] >= fecha)
    ), ["descuento_pct"]]

    if descuento_pct.shape[0] == 1:
        return 1 , descuento_pct.values[0, 0]
    else:
        return 0 , 0
    
    
def recency(recency, q25, q75):
    if recency < pd.Timedelta(days=q25):
        return "RA"
    elif recency <= pd.Timedelta(days=q75):
        return "RB"
    else:
        return "RC"
    
def frequency(cantidad, q25, q75):
    if cantidad > q75:
        return "FA"
    elif cantidad >= q25:
        return "FB"
    else:
        return "FC"
    
    
def monetary(total, q25, q75):
    if total > q75:
        return "MA"
    elif total >= q25:
        return "MB"
    else:
        return "MC"
    
def make_data_to_model(customers, transactions, promotions, date_max):
    # Se genera copia para no afectar los originales
    ctm = customers.copy()
    tst = transactions.copy()
    pmt = promotions.copy()
    
    # Formato a datetime
    pmt["fecha_inicio"] = pd.to_datetime(pmt["fecha_inicio"])
    pmt["fecha_fin"] = pd.to_datetime(pmt["fecha_fin"])
    tst["fecha"] = pd.to_datetime(tst["fecha"])
    pmt["fecha_inicio"] = pd.to_datetime(pmt["fecha_inicio"])
    pmt["fecha_fin"] = pd.to_datetime(pmt["fecha_fin"])
    date_max = pd.to_datetime(date_max)

    # Se corta transaction para train y test
    tst = tst.sort_values("fecha")
    tst_test = tst.loc[tst["fecha"] > date_max].copy()     # test
    tst = tst.loc[tst["fecha"] <= date_max].copy()     # train
    # Label para test
    ctm_test = ctm[["customer_id"]].copy()
    ctm_test["label"] = ctm_test["customer_id"].isin(tst_test["customer_id"].unique()).astype("int")

    # Generacion de columnas
    # total_pagado
    tst["total_pagado"] = tst["precio_unitario"] * tst["cantidad"]
    tst[["es_promo", "descuento_pct"]] = tst.apply(lambda df: es_promo(df["sku"], df["fecha"], pmt), axis=1, result_type='expand')
    tst["total_pagado"] = tst["total_pagado"] * ((100 - tst["descuento_pct"]) / 100)
    # dias de registro
    ctm["dias_de_registro"] = tst["fecha"].max() - pd.to_datetime(ctm["fecha_registro"])
    ctm["dias_de_registro"] = ctm["dias_de_registro"].apply(lambda dt: dt.days)
    
    
    # RMF
    info_ctm_tst = tst.groupby("customer_id").agg(
        recency=("fecha", "max"),
        frequency=("sku", "count"),
        monetary=("total_pagado", "sum"),
        promo=("es_promo", "sum")
    ).reset_index()
    info_ctm_tst['recency'] = tst['fecha'].max() - info_ctm_tst['recency']

    # Recency
    # Se identifca la frecuencia de la cantidad de días que demora un cliente para volver a hacer otra compra
    day_diff = list()
    tst = tst.sort_values(["customer_id", "fecha"]).reset_index(drop=True)
    customer_0 = tst.loc[0, "customer_id"]
    for row in range(tst.shape[0] - 1):
        df_temp = tst.loc[[row, row + 1], ["customer_id", "fecha"]].copy()
        df_temp = df_temp.reset_index(drop=True)
        if df_temp.loc[1, "customer_id"] == customer_0:
            days = df_temp["fecha"].diff()[1].days
            if days > 0: day_diff.append(days)
        customer_0 = df_temp.loc[1, "customer_id"]

    rq25 = np.quantile(day_diff, q=0.25)
    rq75 = np.quantile(day_diff, q=0.75)
    info_ctm_tst["R"] = info_ctm_tst["recency"].apply(lambda r: recency(r, rq25, rq75))

    # Frequency
    fq25 = np.quantile(info_ctm_tst["frequency"].values, q=0.25)
    fq75 = np.quantile(info_ctm_tst["frequency"].values, q=0.75)
    info_ctm_tst["F"] = info_ctm_tst["frequency"].apply(lambda f: frequency(f, fq25, fq75))

    # Monetary
    mq25 = np.quantile(info_ctm_tst["monetary"].values, q=0.25)
    mq75 = np.quantile(info_ctm_tst["monetary"].values, q=0.75)
    info_ctm_tst["M"] = info_ctm_tst["monetary"].apply(lambda m: monetary(m, mq25, mq75))

    # Se calcula el puntaje total
    info_ctm_tst["puntaje_R"] = info_ctm_tst["R"].map(dict(zip(["RA", "RB", "RC"], [1 ,2, 3])))
    info_ctm_tst["puntaje_F"] = info_ctm_tst["F"].map(dict(zip(["FA", "FB", "FC"], [1 ,2, 3])))
    info_ctm_tst["puntaje_M"] = info_ctm_tst["M"].map(dict(zip(["MA", "MB", "MC"], [1 ,2, 3])))
    info_ctm_tst["Puntaje_RFM"] = info_ctm_tst["puntaje_R"] + info_ctm_tst["puntaje_F"] + info_ctm_tst["puntaje_M"]

    mean_rfm = info_ctm_tst["Puntaje_RFM"].mean()
    dsta_rfm = info_ctm_tst["Puntaje_RFM"].std()
    print(f"El puntaje RFM tiene una distribución normal con media: {mean_rfm:,.2f} y desviación estandar: {dsta_rfm:,.2f}.")
    
    # Se incorpora la información de info_ctm_tst a ctm
    ctm = ctm.merge(
        info_ctm_tst[["customer_id", "recency", "frequency", "monetary", "promo", "Puntaje_RFM"]],
        "left", "customer_id"
    )
    ctm["recency"] = ctm["recency"].fillna(ctm["recency"].max())
    ctm["frequency"] = ctm["frequency"].fillna(0)
    ctm["monetary"] = ctm["monetary"].fillna(0)
    ctm["promo"] = ctm["promo"].fillna(0)
    ctm["Puntaje_RFM"] = ctm["Puntaje_RFM"].fillna(0)
    # Arregla el formato de recency a int
    ctm["recency"] = ctm["recency"].apply(lambda days: days.days)
    
    # stats ciudades-rfm
    ciudad_rfm = ctm[ctm["Puntaje_RFM"] > 0].groupby("ciudad").agg(
        ciudad_rfm_mean=("Puntaje_RFM", "mean"),
        cantidad=("Puntaje_RFM", "count"),
        ).reset_index().sort_values("ciudad_rfm_mean", ascending=False).reset_index(drop=True)
    ciudad_rfm["porc"] = ciudad_rfm["cantidad"] / ciudad_rfm["cantidad"].sum()
    # coef
    ciudad_rfm["coef_rfm"] = (ciudad_rfm["ciudad_rfm_mean"] - info_ctm_tst["Puntaje_RFM"].mean()) * (1 - ciudad_rfm["porc"])
    ciudad_rfm["coef_rfm"] = (ciudad_rfm["coef_rfm"] - ciudad_rfm["coef_rfm"].min()) / (ciudad_rfm["coef_rfm"].max() - ciudad_rfm["coef_rfm"].min())

    # stats canales-rfm
    canal_rfm = ctm[ctm["Puntaje_RFM"] > 0].groupby("canal_preferido").agg(
        canal_preferido_rfm_mean=("Puntaje_RFM", "mean"),
        cantidad=("Puntaje_RFM", "count"),
        ).reset_index().sort_values("canal_preferido_rfm_mean", ascending=False).reset_index(drop=True)
    canal_rfm["porc"] = canal_rfm["cantidad"] / canal_rfm["cantidad"].sum()
    # coef
    canal_rfm["coef_rfm"] = (canal_rfm["canal_preferido_rfm_mean"] - info_ctm_tst["Puntaje_RFM"].mean()) * (1 - canal_rfm["porc"])
    canal_rfm["coef_rfm"] = (canal_rfm["coef_rfm"] - canal_rfm["coef_rfm"].min()) / (canal_rfm["coef_rfm"].max() - canal_rfm["coef_rfm"].min())

    # Se agrega la información de ciudades-rfm y canales-rfm a customers
    ctm["ciudad_rfm"] = ctm["ciudad"].map(dict(zip(ciudad_rfm["ciudad"] , ciudad_rfm["coef_rfm"])))
    ctm["canal_rfm"] = ctm["canal_preferido"].map(dict(zip(canal_rfm["canal_preferido"] , canal_rfm["coef_rfm"])))
    
    # Se Normalizan y Escalan variables
    ctm["edad_norm"] = (ctm["edad"].mean() - ctm["edad"]) / ctm["edad"].std()
    ctm["ddr_esca"] = (ctm["dias_de_registro"] - ctm["dias_de_registro"].min()) / (ctm["dias_de_registro"].max() - ctm["dias_de_registro"].min())
    ctm["promo_esca"] = (ctm["promo"] - ctm["promo"].min()) / (ctm["promo"].max() - ctm["promo"].min())
    ctm["punt_norm"] = (ctm["Puntaje_RFM"].mean() - ctm["Puntaje_RFM"]) / ctm["Puntaje_RFM"].std()
    
    return ctm[["edad_norm", "ddr_esca", "promo_esca", "recency", "frequency", "monetary", "punt_norm", "ciudad_rfm", "canal_rfm"]].values , ctm_test["label"].values
    # return ctm[["edad_norm", "ddr_esca", "promo_esca", "punt_norm", "ciudad_rfm", "canal_rfm"]].values , ctm_test["label"].values
    
    
X , y = make_data_to_model(
    customers=customers,
    transactions=transactions,
    promotions=promotions,
    date_max="2025-07-31"
)

y_val = customer_labels["label_recompra_90d"].values
X_val , _ = make_data_to_model(customers, transactions, promotions, date_max='2025-10-31')

np.savetxt("X.csv", X, delimiter=" ")
np.savetxt("y.csv", y, delimiter=" ")
np.savetxt("X_val.csv", X_val, delimiter=" ")
np.savetxt("y_val.csv", y_val, delimiter=" ")