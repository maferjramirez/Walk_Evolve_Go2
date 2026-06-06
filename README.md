# Go2 Evo Walk

Optimizador evolutivo para generar una marcha estable de un cuadrúpedo simulado con PyBullet. El proyecto está pensado para que puedas replicarlo en WSL/Ubuntu 22 o Linux, entrenar sin ventana para obtener resultados más rápido, y reproducir luego el mejor individuo en modo gráfico.

## Qué incluye

- `src/go2_evo_walk/sim.py`: simulación, carga del URDF y función de fitness.
- `src/go2_evo_walk/evolve.py`: entrenamiento evolutivo con DEAP (`go2-evo`).
- `src/go2_evo_walk/replay.py`: reproducción de una corrida guardada (`go2-evo-replay`).
- URDF del Go2 tomado desde el repositorio externo `unitree_ros` de Unitree Robotics.

## Repositorio externo de Unitree

Este proyecto no incluye el código completo de `unitree_ros`. Para obtener el URDF del Go2, clona el repositorio oficial por separado junto a este proyecto:

```bash
git clone https://github.com/unitreerobotics/unitree_ros.git
```

Después podrás usar la ruta:

```bash
unitree_ros/robots/go2_description/urdf/go2_description.urdf
```

## Requisitos

- Python 3.10 o superior.
- `pip` y `venv`.
- En WSL/Ubuntu: soporte gráfico si vas a usar `--gui` (WSLg o X11).
- Dependencias del sistema en Ubuntu/WSL:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip build-essential libgl1-mesa-glx libglib2.0-0
```

## Instalación

Desde la raíz del repositorio:

```bash
cd /mnt/c/Users/dksfd/UNAM/finalEvolutivos
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

Si trabajas con ROS2 y necesitas su entorno, cárgalo antes de ejecutar:

```bash
source /opt/ros/humble/setup.bash
```

## Cómo ejecutar

### Entrenamiento sin GUI

Es la forma más rápida de entrenar. No abre ventanas y es la opción recomendada para correr varias generaciones.

```bash
go2-evo --generations 20 --population 24 --urdf unitree_ros/robots/go2_description/urdf/go2_description.urdf
```

### Entrenamiento con GUI

Abre PyBullet y muestra cada evaluación en pantalla.

```bash
go2-evo --generations 20 --population 24 --gui --urdf unitree_ros/robots/go2_description/urdf/go2_description.urdf
```

### Visualización más lenta para demos

Útil si quieres presentar el comportamiento con mayor claridad.

```bash
go2-evo --generations 10 --population 12 --gui --vis-slowdown 2 --vis-hold --urdf unitree_ros/robots/go2_description/urdf/go2_description.urdf
```

### Guardar la mejor corrida

Por defecto el resultado se guarda como `best_run.json`, pero puedes cambiar la ruta con `--output`.

```bash
go2-evo --generations 50 --population 48 --output best_run.json --urdf unitree_ros/robots/go2_description/urdf/go2_description.urdf
```

### Reproducir una corrida guardada

Después de entrenar, puedes volver a abrir el mejor individuo en GUI:

```bash
go2-evo-replay --input best_run.json --slowdown 3 --hold
```

## Parámetros importantes

- `--urdf`: ruta al URDF del robot. Si clonaste `unitree_ros` junto al proyecto, la ruta recomendada es `unitree_ros/robots/go2_description/urdf/go2_description.urdf`.
- `--gui`: activa PyBullet con ventana gráfica.
- `--population`: tamaño de la población.
- `--generations`: número de generaciones.
- `--episode-seconds`: tiempo simulado por evaluación.
- `--settle-seconds`: tiempo de asentamiento antes de medir la marcha.
- `--crossover` y `--mutation`: probabilidades de cruce y mutación.
- `--elitism`: cantidad de individuos élite que se conservan entre generaciones.
- `--output`: archivo JSON donde se guarda el mejor individuo.
- `--vis-slowdown` y `--vis-hold`: controlan la reproducción visual cuando usas `--gui`.

## Resultados esperados

Al terminar, el programa imprime un resumen en consola y guarda un JSON con el mejor individuo. Ese archivo contiene:

- `fitness`: mejor valor de aptitud encontrado.
- `genome`: parámetros evolutivos del controlador.
- `gene_count`: número de genes usados.
- `urdf`: URDF empleado.
- `generations` y `population`: configuración de la corrida.

## Flujo reproducible recomendado

1. Instala dependencias.
2. Ejecuta el entrenamiento sin GUI para obtener el mejor individuo.
3. Reproduce el resultado con `go2-evo-replay` para verificar visualmente el comportamiento.
4. Si necesitas una demostración más clara, usa `--vis-slowdown` y `--vis-hold`.

## Problemas comunes

- Si VS Code marca imports como no resueltos, verifica que el entorno virtual esté activado en el editor.
- Si no aparece la GUI, confirma que tu entorno tiene soporte gráfico (WSLg o X11).
- Si el URDF no se encuentra, verifica que `unitree_ros` esté clonado aparte y que la ruta apunte al archivo `go2_description.urdf` correcto.

## Siguiente paso sugerido

Si quieres llevar esto a un proyecto más completo, el siguiente paso sería conectar el controlador a ROS2/Gazebo y probar la misma política sobre el robot o un entorno más realista.

