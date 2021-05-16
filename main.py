import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import math
from matplotlib.animation import PillowWriter
import random
from scipy.stats import norm
import base64
from io import BytesIO


# Clase particula, recibe las probabilidades de contagio, el tiempo de simulación, tiempo que dura el virus,
# Probabilidad de llevar mascarilla y probabilidad de iniciar contagiado
class Particle:
    # Tamaño de la particula
    size = 10

    # t = tiempo de simulación
    # point = ax
    # p_mask = Probabilidad de llevar mascarilla
    # p_infected = Probabilidad de iniciar contagiado
    # p_c_mask_not = Probabilidad de contagio si la particula contagiada tiene mascarilla y demás no
    # p_c_mask_mask = Probabilidad de contagio si ambas particulas tienen mascarilla
    # p_c_not_mask = Probabilidad de contagio si la particula contagiada no tiene mascarilla y demás si
    # p_c_not_not = Probabilidad de contagio si ambas particulas no tienen mascarilla
    # time_infected = Tiempo que dura el virus

    def __init__(self, t, point, p_mask, p_infected, p_c_mask_not, p_c_mask_mask, p_c_not_mask, p_c_not_not,
                 time_infected):

        # Array de tamaño 't'(Tiempo de simulación) en la cual se guardarán las posiciones 'x'
        # Llena cada array con ceros.
        self.x = np.zeros(t)
        # Array de tamaño 't'(Tiempo de simulación) en la cual se guardarán las posiciones 'y'
        self.y = np.zeros(t)

        self.p_c_mask_not = p_c_mask_not
        self.p_c_mask_mask = p_c_mask_mask
        self.p_c_not_mask = p_c_not_mask
        self.p_c_not_not = p_c_not_not

        # Guarda si la particula se vuelve immune
        self.immune = False

        # Guarda si la particula muere
        self.isDead = False

        # Guarda si la particula se infectó en la simulación
        self.hasInfected = False

        # Contador que comienza a aumentar cuando una particula se infecta
        # (Sirve para compararlo con el tiempo que dura el virus)
        self.count_infected = 0

        self.timeInfected = time_infected

        # Posición inicial entre 0 y 100 en 'x'
        self.x[0] = random.uniform(0, 100)

        # Posición inicial entre 0 y 100 en 'y'
        self.y[0] = random.uniform(0, 100)

        # Velocidad que tendrá la particula, cambia cada vez que cambia de dirección
        self.b = random.uniform(0.1, 0.5)

        self.point = point

        # Posición final en 'x' y 'y', cambia cada vez que la particula llega a esa coordenada
        self.xFinal = random.uniform(0, 100)
        self.yFinal = random.uniform(0, 100)

        # Radio de posición final: 2 en x
        self.maxX = self.xFinal + 2
        self.minX = self.xFinal - 2

        # Radio de posición final: 2 en y
        self.maxY = self.yFinal + 2
        self.minY = self.yFinal - 2

        # Mueve la particula dependiendo la posición final
        self.move(t)

        # Guarda si la particula inicia infectada
        self.infected = np.random.random() <= p_infected

        # Guarda si la particula tiene mascarilla
        self.mask = np.random.random() <= p_mask
        self.circle, = self.point.plot([], [], lw=2, marker="o", color="red", alpha=0,
                                       markersize=self.size * 3)
        if self.infected:
            self.line, = point.plot([], [], lw=2, marker="o", color="red")
            self.circle.set_alpha(0.5)

        else:
            self.line, = point.plot([], [], lw=2, marker="o", color="blue")

            if self.mask:
                self.line.set_color("green")

    def move(self, t):

        # Cambia la posición de la particula 't' veces, dependiendo el tiempo de simulación
        for i in range(1, t):

            # Genera el ángulo de dirección hacia el punto final.
            angle = math.atan2(self.yFinal - self.y[i - 1], self.xFinal - self.x[i - 1])
            vx = self.b * np.cos(angle)
            vy = self.b * np.sin(angle)
            self.x[i] = (self.x[i - 1] + vx) % 100
            self.y[i] = (self.y[i - 1] + vy) % 100

            # Si la particula está dentro del radio de la coordenada final
            # Cambia de dirección y de coordenada final
            if (self.maxX >= self.x[i] >= self.minX) and (self.maxY >= self.y[i] >= self.minY):
                self.b = random.uniform(0.1, 0.5)
                self.xFinal = random.uniform(0, 100)
                self.yFinal = random.uniform(0, 100)

                self.maxX = self.xFinal + 1
                self.minX = self.xFinal - 1

                self.maxY = self.yFinal + 1
                self.minY = self.yFinal - 1

    # Si la particula está contagiada, verifica si las demás partículas están en el radio de la partícula contagiada
    # De igual forma toma en cuenta las probabilidades de llevar mascarilla, tanto del infectado, como de los demás.
    # Cambia visualmente si la partícula se contagia.
    def isInto(self, particles, i):

        for x in particles:
            if (x.isDead is False and x.immune is False) and x.infected is False and x != self and np.sqrt(
                    (self.x[i] - x.x[i]) ** 2 + (self.y[i] - x.y[i]) ** 2) <= 2.5:
                if x.mask:
                    if (np.random.random() <= self.p_c_mask_mask and self.mask) or (
                            np.random.random() <= self.p_c_mask_not and self.mask is False):
                        x.hasInfected = True
                        x.infected = True
                        x.circle.set_alpha(0.5)
                        x.count_infected = 0
                else:
                    if (np.random.random() <= self.p_c_not_mask and self.mask) or (
                            np.random.random() <= self.p_c_not_not and self.mask is False):
                        x.hasInfected = True
                        x.infected = True
                        x.circle.set_alpha(0.5)
                        x.line.set_color("red")
                        x.count_infected = 0


# Genera un objecto partícula
def getParticle(t, ax, p_mask, p_infected, p_c_mask_not, p_c_mask_mask, p_c_not_mask, p_c_not_not, time_infected):
    return Particle(t, ax, p_mask, p_infected, p_c_mask_not, p_c_mask_mask, p_c_not_mask, p_c_not_not, time_infected)


# Genera un array de partículas, dependiendo el tiempo de simulación y el número de partículas
# Genera tiempos que duraría el virus en cada particula, con una media y desviación específica
def getParticles(n, t, ax, p_mask, p_infected, p_c_mask_not, p_c_mask_mask, p_c_not_mask, p_c_not_not, mu, sigma):
    particles = []
    time_infected = norm.ppf(np.random.random(n), loc=mu, scale=sigma)
    for x in range(n):
        particles.append(
            getParticle(t, ax, p_mask, p_infected, p_c_mask_not, p_c_mask_mask, p_c_not_mask, p_c_not_not,
                        int(time_infected[x])))
    return np.array(particles, dtype="object")


# Clase general la cual guarda las partículas inmunes, muertas y los gráficos dados en simulación
class Simulation:
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(xlim=(0, 100), ylim=(0, 100))
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, color="white")
    infected_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color="white")
    dead_text = ax.text(0.02, 0.92, '', transform=ax.transAxes, color="white")
    recovered_text = ax.text(0.02, 0.89, '', transform=ax.transAxes, color="white")
    infected2_text = ax.text(0.02, 0.86, '', transform=ax.transAxes, color="white")
    health_text = ax.text(0.02, 0.83, '', transform=ax.transAxes, color="white")
    ax.set_facecolor('black')

    # Contador de partículas inmunes
    recovery = 0

    # Contador de partículas muertas
    deaths = 0
    writer = PillowWriter(fps=25)

    # Figura que muestra las particulas, susceptibles, infectadas, muertas y recuperadas con inmunidad
    f = plt.figure(2)

    ax2 = f.subplots()
    ax2.set_facecolor('#dddddd')

    # Figura que muestra el porcentaje de particulas contagiadas con mascarilla y las que no.
    fm = plt.figure(3)

    ax3 = fm.subplots()
    ax3.set_facecolor('#dddddd')
    ax3.set_ylim(0, 100)

    # Array de partículas susceptibles
    y_rand = []

    # Array de partículas muertas
    y_rand_dead = []

    # Array de partículas inmunes
    y_rand_immune = []

    # Array de partículas infectadas
    y_infects = []

    # Array de particulas infectadas sin mascarilla
    y_no_mask_infected = []

    # Array de particulas infectadas con mascarilla
    y_mask_infected = []

    # n = número de partículas
    # t = tiempo de simulación
    # p_mask = Probabilidad de llevar mascarilla
    # p_infected = Probabilidad de iniciar contagiado
    # p_c_mask_not = Probabilidad de contagio si la particula contagiada tiene mascarilla y demás no
    # p_c_mask_mask = Probabilidad de contagio si ambas particulas tienen mascarilla
    # p_c_not_mask = Probabilidad de contagio si la particula contagiada no tiene mascarilla y demás si
    # p_c_not_not = Probabilidad de contagio si ambas particulas no tienen mascarilla
    # p_dead = Probabilidad de muerte de una partícula, dado un contagio
    # mu y sigma = Media y desviación respecto al tiempo que dura el virus en cada partícula
    def __init__(self, n, t, p_mask, p_infected, p_c_mask_not, p_c_mask_mask, p_c_not_mask, p_c_not_not, p_dead, mu,
                 sigma):
        self.n = n
        self.t = t
        self.particles = getParticles(n, t, self.ax, p_mask, p_infected, p_c_mask_not, p_c_mask_mask, p_c_not_mask,
                                      p_c_not_not, mu, sigma)
        self.p_dead = p_dead

        # Guarda el número de partículas con mascarilla
        self.count_mask = sum(map(lambda z: z.mask, self.particles))

        # Guarda el número de partículas sin mascarilla
        self.count_not_mask = self.n - self.count_mask

    # Anima todas las partículas, dependiendo las posiciones de cada una en cada instante de tiempo
    def animate(self, i):

        self.time_text.set_text('Tiempo = %.0f' % i + '/%.0f' % self.t)

        for x in self.particles:
            x.line.set_data(x.x[i], x.y[i])

            if x.infected:
                x.count_infected += 1

                # Si la partícula cumple con el tiempo de duración del virus, deja de estar infectada y
                # Dado un número generado, cambia el estado de la partícula a muerta, o recuperada con inmunidad
                if x.count_infected == x.timeInfected:
                    x.infected = False
                    if np.random.random() <= self.p_dead:
                        x.isDead = True
                        self.deaths += 1
                        x.line.set_color("gray")
                        self.dead_text.set_text('Muertos = %.0f' % self.deaths)
                    else:
                        self.recovery += 1
                        x.immune = True
                        x.line.set_color("yellow")
                        self.recovered_text.set_text('Inmunes = %.0f' % self.recovery)
                    x.circle.set_alpha(0)
                else:
                    x.circle.set_data(x.x[i], x.y[i])
                    x.isInto(self.particles, i)

            # Número de infectados, disminuye si alguno se recupera o muere, y aumenta si se contagia
            count_initial = sum(map(lambda z: z.infected, self.particles))
            self.infected_text.set_text('Infectados = %.0f' % count_initial)

            # Número de infectados totales en simulación
            count = sum(map(lambda z: z.hasInfected, self.particles))
            self.infected2_text.set_text('Infectados (Hoy) = %.0f' % count)

            # Número de partículas susceptibles
            count_health = sum(map(lambda z: (z.infected
                                              is False) and (z.isDead
                                                             is False) and (z.immune
                                                                            is False), self.particles))
            self.health_text.set_text('Susceptibles = %.0f' % count_health)

        self.y_rand.append(sum(map(lambda z: (z.infected
                                              is False) and (z.isDead
                                                             is False) and (z.immune
                                                                            is False), self.particles)))
        self.y_infects.append(sum(map(lambda z: z.infected, self.particles)))
        self.y_rand_dead.append(self.deaths)
        self.y_rand_immune.append(self.recovery)

        self.y_no_mask_infected.append(
            sum(map(lambda z: z.mask is False and z.hasInfected, self.particles)) * 100 / self.count_not_mask)
        self.y_mask_infected.append(sum(
            map(lambda z: z.mask and z.hasInfected, self.particles)) * 100 / self.count_mask)

    def getMaskReport(self):
        self.ax3.plot(self.y_no_mask_infected, color="blue", label='% Personas sin máscara contagiadas', lw=2)
        self.ax3.plot(self.y_mask_infected, color="green", label='% Personas con máscara contagiadas', lw=2)

        self.ax3.set_xlabel('Tiempo / días')
        self.ax3.set_ylabel('Personas %')
        self.ax3.grid(b=True, which='major', c='w', lw=2, ls='-')
        self.ax3.legend()

        buf = BytesIO()
        self.fm.savefig(buf, format="png")

        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        return data

    def getReportGeneral(self):
        self.ax2.plot(self.y_rand, color="blue", label='Susceptibles', lw=2)
        self.ax2.plot(self.y_infects, color="green", label='Infectados', lw=2)
        self.ax2.plot(self.y_rand_dead, color="red", label='Muertos', lw=2)
        self.ax2.plot(self.y_rand_immune, color="yellow", label='Recuperados con inmunidad', lw=2)
        self.ax2.set_xlabel('Tiempo / días')
        self.ax2.set_ylabel('Personas')
        self.ax2.grid(b=True, which='major', c='w', lw=2, ls='-')
        self.ax2.legend()

        buf = BytesIO()
        self.f.savefig(buf, format="png")

        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        return data

    def gif(self):
        str_gif = "virusSimulation.gif"
        with open(str_gif, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("ascii")

    def simulate(self):
        print("Cargando...")
        count_initial = sum(map(lambda x: x.infected, self.particles))
        print("Infectados iniciales: ", count_initial)

        self.time_text.set_text('Tiempo = 0/%.0f' % self.t)
        self.dead_text.set_text('Muertos = 0')
        self.recovered_text.set_text('Inmunes = 0')

        anim = animation.FuncAnimation(self.fig, self.animate, self.t, interval=10000 / self.t, blit=False)
        anim.save("virusSimulation.gif", writer=self.writer)

        print("Recuperados: ", self.recovery)
        print("Muertos: ", self.deaths)

        count = sum(map(lambda x: x.hasInfected, self.particles))
        print("Infectados: ", count)

        count_mask = sum(map(lambda z: z.mask, self.particles))
        print("Personas con máscara: ", count_mask)

        count_not_mask = self.n - count_mask
        print("Personas sin máscara: ", count_not_mask)

        count_mask_infected = sum(map(lambda z: z.mask and z.hasInfected, self.particles))
        print("Personas con máscara infectadas: ", count_mask_infected)

        count_not_mask_infected = sum(map(lambda z: z.mask is False and z.hasInfected, self.particles))
        print("Personas sin máscara infectadas: ", count_not_mask_infected)


simulation = Simulation(200, 1000, p_mask=0.8, p_infected=0.1, p_c_mask_not=0.1, p_c_mask_mask=0.01, p_c_not_mask=0.15,
                        p_c_not_not=0.3, p_dead=0.2, mu=200, sigma=50)

# Inicia la simulación
simulation.simulate()

# Retorna el gif(base64) de la simulación
simulation.gif()

# Retorna la imágen(base64) del reporte de partículas infectadas con mascarilla y sin ella
simulation.getMaskReport()

# Retorna la imagen(base64) del reporte general(Susceptibles, infectados, muertos y recuperados con inmuidad)
simulation.getReportGeneral()
