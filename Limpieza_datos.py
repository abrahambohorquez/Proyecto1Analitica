#Descargue localmente la base de datos de cordoba ya que 
#Github no permite subir archivos de más de 100 MB

#Importe pandas
import pandas as pd
import unicodedata

#Indique el path local al cual esta desgargo la base de datos
#ASEGURESE DE CAMBIAR EL PATH A DONDE QUIERE IMPORTAR EL ARCHIVO
path = "C:\\Users\\lucas\\Desktop\\Uniandes\\Industrial\\8vo semestre\\Analitica Computacional\\Proyecto 1\\Resultados_ICFES_Cordoba_raw.csv"
df = pd.read_csv(path, encoding="utf-8-sig",low_memory=False)
names = df.columns
#Verificar los data types y asegurarse que todo este en orden
print(df.dtypes)

#Se encontro que las fechas de naciomiento de los estudiantes estan en formato string
#Para limpieza será mejor pasarlas a formato datetime
df["estu_fechanacimiento"] = pd.to_datetime(df["estu_fechanacimiento"],format="%d/%m/%Y",errors="coerce")

#Se eliminan valores duplicados en las 52 columnas

df = df.drop_duplicates()

#Posteriormente se separan las variables string para ver si hay inconsistencias en ellas

string = []
for i in names:
    if df[i].dtype == "str":
        string.append(i)
print(string)

#Se imprimen los valores unicos de cada string para ver si hay strings inconsistentes o raros
#Adicionalmente se verufuca que para la categoria "cole_depto_ubicacion" solo haya un
#unico argumento que sea 'CORDOBA' indicando una propia importación de los datos
for i in string:
    print(i,df[i].unique())

#Se encuentra una inconsistencia en categoria "cole_mcpio_ubicacion", hay municipios identicos que aparecen sin y con acentos
#Por ejemplo aparece Montería y Monteria como datos separados


def remove_accents(text):
    if pd.isna(text):
        return text
    return ''.join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )

df["ubicacion_norm"] = df["cole_mcpio_ubicacion"].apply(remove_accents)

correcciones = {
    "MONTERIA": "MONTERÍA",
    "CERETE": "CERETÉ",
    "CHIMA": "CHIMÁ",
    "CHINU": "CHINÚ",
    "SAHAGUN": "SAHAGÚN",
    "LOS CORDOBAS": "LOS CÓRDOBAS",
    "PURISIMA": "PURÍSIMA",
    "CIENAGA DE ORO": "CIÉNAGA DE ORO",
    "SAN JOSE DE URE": "SAN JOSÉ DE URÉ",
    "SAN ANDRES SOTAVENTO": "SAN ANDRÉS DE SOTAVENTO"
}

df["cole_mcpio_ubicacion"] = (
    df["cole_mcpio_ubicacion"]
    .apply(lambda x: correcciones.get(remove_accents(x), x))
)
df.drop(columns=["ubicacion_norm"], inplace=True)

#Las demás inconsistencias se resolvieron cambiando el encoding a "utf-8-sig" para que
#la pagina leyera los datos de manera apropiada
#Cabe recalcar que la categoria "cole_nombre_sede" registro de manera erronea los datos
#Haciendo que varios caracteres con tildes se reemplacen por ¿ y sea imposible
#recuperar el caracter original

#Como pasos generales eliminamos espacios en iniciales y finales en datos:
for i in string:
    df[i] = (
        df[i]
        .str.strip()
        .str.upper()
    )

#Ya con los strings propiamente leidos y modificados se puede crear el nuevo set de datos
#ASEGURESE DE CAMBIAR EL PATH A DONDE QUIERE GUARDAR EL ARCHIVO
path_save = "C:\\Users\\lucas\\Desktop\\Uniandes\\Industrial\\8vo semestre\\Analitica Computacional\\Proyecto 1\\Resultados_ICFES_Cordoba_clean.csv"
df.to_csv(path_save,index=False,encoding="utf-8-sig"
)

















