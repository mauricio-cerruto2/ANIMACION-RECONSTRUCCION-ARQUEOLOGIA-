# ANIMACION-RECONSTRUCCION-ARQUEOLOGIA-
HERLAN MAURICIO CERRUTO SALAZAR    C.I:10954648


import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from scipy.interpolate import CubicSpline
from matplotlib.animation import FuncAnimation, PillowWriter


# ==========================================
# 1. CONFIGURACION DEL PROYECTO
# ==========================================

RUTA_IMAGEN = "vasija_tiwanaku.png"
CARPETA = "resultados"

os.makedirs(CARPETA, exist_ok=True)

imagen = Image.open(RUTA_IMAGEN).convert("RGB")
img = np.array(imagen)

alto, ancho = img.shape[:2]

print("RECONSTRUCCION ARQUEOLOGICA - TIWANAKU")
print("---------------------------------------")


# ==========================================
# 2. SELECCIONAR EL EJE CENTRAL
# ==========================================

fig, ax = plt.subplots(figsize=(8, 9))
ax.imshow(img)

ax.set_title(
    "PASO 1: Selecciona el eje central\n"
    "Primero el centro del borde superior y luego el de la base"
)

print("\nSelecciona dos puntos sobre el centro vertical de la vasija.")
print("Primero el centro superior y despues el centro inferior.")

eje_puntos = plt.ginput(2, timeout=0)

if len(eje_puntos) != 2:
    plt.close()
    raise ValueError("Debes seleccionar los dos puntos del eje.")

eje_x = np.mean([p[0] for p in eje_puntos])

ax.axvline(eje_x, color="yellow", linestyle="--")
ax.set_title("Eje central seleccionado")
plt.draw()

print("\nEje central establecido.")


# ==========================================
# 3. SELECCIONAR EL PERFIL DE LA VASIJA
# ==========================================

print("\nAhora selecciona el perfil DERECHO de la vasija.")
print("Empieza en el borde superior y termina en la base.")
print("No incluyas las asas laterales.")
print("Presiona ENTER cuando termines.")

ax.set_title(
    "PASO 2: Selecciona el perfil derecho\n"
    "Desde el borde superior hasta la base. ENTER para terminar"
)

perfil = plt.ginput(n=-1, timeout=0)
plt.close()

if len(perfil) < 4:
    raise ValueError("Selecciona al menos cuatro puntos del perfil.")

perfil = np.array(perfil)

# Coordenadas verticales y radios
y_original = perfil[:, 1]
radios_original = perfil[:, 0] - eje_x

# Ordenar desde arriba hacia abajo
orden = np.argsort(y_original)
y_original = y_original[orden]
radios_original = radios_original[orden]

# Evitar radios negativos
radios_original = np.maximum(radios_original, 0)

# Eliminar coordenadas Y repetidas
y_unicos, indices = np.unique(y_original, return_inverse=True)

radios_unicos = np.bincount(
    indices,
    weights=radios_original
) / np.bincount(indices)

if len(y_unicos) < 4:
    raise ValueError("Necesitas al menos cuatro alturas diferentes.")

# Guardar coordenadas
np.savetxt(
    f"{CARPETA}/perfil_coordenadas.csv",
    np.column_stack((y_unicos, radios_unicos)),
    delimiter=",",
    header="Y,Radio",
    comments=""
)

print(f"\nSe seleccionaron {len(perfil)} puntos.")
print("Coordenadas guardadas.")


# ==========================================
# 4. CALCULAR EL SPLINE CUBICO
# ==========================================

spline = CubicSpline(
    y_unicos,
    radios_unicos,
    bc_type="natural"
)

y_suave = np.linspace(
    y_unicos.min(),
    y_unicos.max(),
    150
)

radio_suave = np.maximum(spline(y_suave), 0)

print("\nSpline cubico calculado correctamente.")


# ==========================================
# 5. GRAFICO DEL SPLINE Y COMPARACION
# ==========================================

x_derecha = eje_x + radio_suave
x_izquierda = eje_x - radio_suave

fig, axes = plt.subplots(1, 2, figsize=(13, 8))

# Reconstruccion del contorno
axes[0].plot(
    x_derecha,
    y_suave,
    color="blue",
    linewidth=2,
    label="Spline cubico"
)

axes[0].plot(
    x_izquierda,
    y_suave,
    color="blue",
    linewidth=2
)

axes[0].scatter(
    eje_x + radios_unicos,
    y_unicos,
    color="red",
    s=18,
    label="Puntos de control"
)

axes[0].scatter(
    eje_x - radios_unicos,
    y_unicos,
    color="red",
    s=18
)

axes[0].invert_yaxis()
axes[0].set_aspect("equal")
axes[0].set_title("Reconstruccion mediante Splines")
axes[0].set_xlabel("Coordenada X")
axes[0].set_ylabel("Coordenada Y")
axes[0].grid()
axes[0].legend()

# Fotografia original
axes[1].imshow(img)
axes[1].set_title("Vasija original")
axes[1].axis("off")

plt.tight_layout()
plt.savefig(f"{CARPETA}/comparacion.png", dpi=150)
plt.show()


# ==========================================
# 6. CONSTRUIR LA SUPERFICIE 3D
# ==========================================

print("\nConstruyendo modelo tridimensional...")

theta = np.linspace(0, 2 * np.pi, 120)

# Coordenadas de la superficie
T, Y = np.meshgrid(theta, y_suave)

R = np.tile(radio_suave[:, None], (1, len(theta)))

X = R * np.cos(T)
Z = R * np.sin(T)

# Altura del modelo
ALTURA = y_suave.max() - Y

# Obtener textura aproximada desde la fotografia
columnas = np.clip(
    np.rint(eje_x + R * np.cos(T)).astype(int),
    0,
    ancho - 1
)

filas = np.clip(
    np.rint(Y).astype(int),
    0,
    alto - 1
)

textura = img[filas, columnas] / 255.0

# Convertir a RGBA
colores = np.concatenate(
    [textura, np.ones((*textura.shape[:2], 1))],
    axis=2
)


# ==========================================
# 7. FUNCION PARA DIBUJAR EL MODELO
# ==========================================

def dibujar_modelo(ax, colores_actuales):

    ax.plot_surface(
        X,
        Z,
        ALTURA,
        facecolors=colores_actuales[:-1, :-1],
        linewidth=0,
        antialiased=True,
        shade=False
    )

    ax.set_title("Vasija de Tiwanaku - Modelo 3D")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Altura")

    ax.set_box_aspect((1, 1, 1.5))
    ax.view_init(elev=22, azim=35)

    ax.set_axis_off()


# Guardar modelo 3D inicial
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")

dibujar_modelo(ax, colores)

plt.savefig(
    f"{CARPETA}/modelo_3D.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# 8. ANIMACION DE ROTACION 360 GRADOS
# ==========================================



# ==========================================
# 8. ANIMACION 3D DE ROTACION 360 GRADOS
# ==========================================

print("\nGenerando animacion 3D...")

fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection="3d")

def actualizar(frame):

    ax.clear()

    # Dibujar la superficie 3D
    ax.plot_surface(
        X,
        Z,
        ALTURA,
        facecolors=colores[:-1, :-1],
        linewidth=0,
        antialiased=True,
        shade=False
    )

    # Rotar la vista alrededor de la vasija
    angulo = frame * 5

    ax.view_init(
        elev=22,
        azim=angulo
    )

    ax.set_title(
        f"Vasija de Tiwanaku - {angulo} grados"
    )

    ax.set_box_aspect((1, 1, 1.5))
    ax.set_axis_off()

    return []

# Crear animacion
animacion = FuncAnimation(
    fig,
    actualizar,
    frames=72,
    interval=80,
    repeat=True,
    blit=False
)

# Guardar GIF
ruta_gif = f"{CARPETA}/vasija_rotando.gif"

animacion.save(
    ruta_gif,
    writer=PillowWriter(fps=15)
)

print("GIF guardado en:", ruta_gif)

# Mostrar animacion en pantalla
plt.show()



# ==========================================
# 9. FINALIZAR
# ==========================================

print("\n===================================")
print("PROYECTO FINALIZADO CORRECTAMENTE")
print("===================================")

print("Archivos generados:")

print("- comparacion.png")
print("- perfil_coordenadas.csv")
print("- modelo_3D.png")
print("- vasija_rotando.gif")

print("\nRevisa la carpeta resultados.")
