const tf = require('@tensorflow/tfjs');
const { Sequential, Dense } = require('@tensorflow/tfjs-layers');
const { Adam } = require('@tensorflow/tfjs-optimizers');
const mt = require('mathjs');



let StudentState = {
    "C1": {
        "PorcentajeDeAvance": 60,
        "valueDQN": new Set(),
        "N1C1": {
            "PorcentajeDeAvance": 30,
            "valueDQN": new Set(),
            "F1N1C1": {
                "PorcentajeDeAvance": 15,
                "valueDQN": new Set(),
                "H1F1N1C1": {
                    "PorcentajeDeAvance": 15,
                    "valueDQN": new Set()
                }
            },
            "F2N1C1": {
                "PorcentajeDeAvance": 45,
                "valueDQN": new Set(),
                "H2F2N1C1": {
                    "PorcentajeDeAvance": 45,
                    "valueDQN": new Set()
                }
            }
        },
        "N2C1": {
            "PorcentajeDeAvance": 90,
            "valueDQN": new Set(),
            "F1N2C1": {
                "PorcentajeDeAvance": 100,
                "valueDQN": new Set(),
                "H1F1N2C1": {
                    "PorcentajeDeAvance": 100,
                    "valueDQN": new Set()
                }
            },
            "F2N2C1": {
                "PorcentajeDeAvance": 80,
                "valueDQN": new Set(),
                "H2F2N2C1": {
                    "PorcentajeDeAvance": 80,
                    "valueDQN": new Set()
                }
            }
        }
    },
    "C2": {
        "PorcentajeDeAvance": 50,
        "valueDQN": new Set(),
        "N1C2": {
            "PorcentajeDeAvance": 80,
            "valueDQN": new Set(),
            "F1N1C2": {
                "PorcentajeDeAvance": 80,
                "valueDQN": new Set(),
                "H1F1N1C2": {
                    "PorcentajeDeAvance": 80,
                    "valueDQN": new Set()
                }
            },
            "F2N1C2": {
                "PorcentajeDeAvance": 80,
                "valueDQN": new Set(),
                "H2F2N1C2": {
                    "PorcentajeDeAvance": 80,
                    "valueDQN": new Set()
                }
            }
        },
        "N2C2": {
            "PorcentajeDeAvance": 20,
            "valueDQN": new Set(),
            "F1N2C2": {
                "PorcentajeDeAvance": 10,
                "valueDQN": new Set(),
                "H1F1N2C2": {
                    "PorcentajeDeAvance": 10,
                    "valueDQN": new Set()
                }
            },
            "F2N2C2": {
                "PorcentajeDeAvance": 30,
                "valueDQN": new Set(),
                "H2F2N2C2": {
                    "PorcentajeDeAvance": 30,
                    "valueDQN": new Set()
                }
            }
        }
    }
};


// Creación de la clase DEEP Q-LEARNING
class DQN {
    // El método constructor de la clase DQN inicializa los atributos y construye el modelo para el aprendizaje.
    constructor(stateSize, actionSize) {
        this.stateSize = stateSize;
        this.actionSize = actionSize;
        this.memory = new deque(); // No hay un equivalente directo de 'deque' en JavaScript, puedes usar un array en su lugar.
        this.gamma = 0.95;
        this.epsilon = 1.0;
        this.epsilonMin = 0.01;
        this.epsilonDecay = 0.995;
        this.learningRate = 0.001;
        this.model = this._buildModel();
    }

    // El método _buildModel de la clase DQN crea y compila el modelo utilizado para el aprendizaje.
    _buildModel() {
        const model = tf.sequential();
        model.add(tf.layers.dense({ units: 66, inputShape: [this.stateSize], activation: 'relu' }));
        model.add(tf.layers.dense({ units: 24, activation: 'relu' }));
        model.add(tf.layers.dense({ units: this.actionSize, activation: 'linear' }));
        model.compile({ loss: 'meanSquaredError', optimizer: new Adam({ learningRate: this.learningRate }) });
        return model;
    }

    // El método remember de la clase DQN almacena la experiencia pasada en la memoria de reproducción para su posterior uso en el entrenamiento del modelo.
    remember(state, reward, nextState) {
        this.memory.push([state, reward, nextState]);
    }

    // Función importante donde se realiza la predicción de la red neuronal, donde se define el peso que tendrá cada competencia, nivel, fase y habilidad.
    predict(state) {
        // Extraer los porcentajes del estado y convertirlos a decimal
        const percentages = state.map(item => item.porcentaje / 100);

        // Convertir los valores porcentuales en un array 1D de TensorFlow.js
        const percentageValues = tf.tensor2d([percentages]);

        // Imprimir el estado y los porcentajes
        console.log("Estado para predecir:", percentageValues.arraySync());
        console.log("Porcentajes:", percentages);

        // Predecir los valores de acción utilizando el modelo
        const predictValues = this.model.predict(percentageValues);

        // La variable predictValues contiene los resultados de la predicción de la red neuronal, se necesitan resultados positivos.
        // Calcular el valor absoluto de los valores del resultado de la predicción
        const absoluteValues = predictValues.abs().arraySync();

        // Calcular el peso de cada clave (Competencia, nivel, fase y habilidad) multiplicando los valores absolutos por los porcentajes
        const weight = mt.dotMultiply(absoluteValues, percentages);

        // Imprimir información
        console.log("Resultado de predicción:", predictValues.arraySync());
        console.log("Resultado de predicción con los valores absolutos", absoluteValues);
        console.log("Pesos:", weight);

        return weight;
    }
}

//-------------- FIN DE LA CLASE DQN --------------------------




// Función para crear el estado utilizando el estado del alumno
function crearState(objeto, padre = null) {
    // Inicializa una lista para almacenar los elementos del estado
    const state = [];
    // Recorre cada clave y valor en el objeto que representa el estado
    Object.entries(objeto).forEach(([clave, valor]) => {
        // Verifica si el valor es un objeto (es decir, un nivel más profundo del estado)
        if (typeof valor === 'object' && valor !== null) {
            // Obtiene el porcentaje de avance del nivel actual
            const porcentaje = valor.PorcentajeDeAvance;
            // Añade un objeto con el nombre y porcentaje de avance del nivel actual a la lista de estado
            state.push({ name: clave, porcentaje });
            // Llama recursivamente a la función para explorar los niveles más profundos del estado
            state.push(...crearState(valor, clave));
        }
    });
    // Retorna la lista completa de objetos que representan el estado
    return state;
}



// Crear estado utilizando el estado del alumno
const state = crearState(StudentState);
console.log(state);

// Calcular tamaño o longitud del estado para tener el número de predicciones que hará la red neuronal
const stateSize = state.length;
const actionSize = state.length;

// Inicialización del agente DQN
const agent = new DQN(stateSize, actionSize);
const result = agent.predict(state); // Utilizar la función predict del agente DQN



const p = []; // p es una manera de mostrar organizadamente los resultados de la predicción junto con su respectiva clave, sea C, N, F y H

state.forEach((item, i) => {
    p.push({
        name: item.name,
        result: result[0][i]
    });
});

console.log("estado del alumno", StudentState);
console.log("result_object", p);


// Convertir el array p en un objeto para facilitar el acceso a los valores
const pDict = p.reduce((acc, item) => {
    acc[item.name] = item.result;
    return acc;
}, {});

console.log(pDict);


// Función para asignarle el peso(Resultado de la predicción) a cada competencia, nivel, fase y habilidad
function assignDQNValues(objeto) {
    for (const clave in objeto) {
        if (typeof objeto[clave] === 'object' && objeto[clave] !== null) {
            if (clave in pDict) {
                objeto[clave].valueDQN = pDict[clave];
            }
            assignDQNValues(objeto[clave]);
        }
    }
}

// Llamar a la función para asignar los valores de DQN
assignDQNValues(StudentState);

// Imprimir StudentState actualizado
console.log("estado del alumno con valores dqn", StudentState);


// Función para crear la ruta
function createPath(objeto) {
    let maxPath = [];
    let maxPercentage = 0;

    for (const key in objeto) {
        if (typeof objeto[key] === 'object' && objeto[key] !== null) {
            const percentage = objeto[key].PorcentajeDeAvance || 0;
            if (percentage < 100) { // Verificar si el porcentaje no está completo
                if (percentage > maxPercentage) {
                    maxPercentage = percentage;
                    maxPath = [key];
                } else if (percentage === maxPercentage) {
                    maxPath.push(key);
                }
            }
        }
    }

    if (maxPath.length > 0) {
        const firstKey = maxPath[0];
        const nextLevel = objeto[firstKey];
        maxPath.push(...createPath(nextLevel));
    }

    return maxPath;
}

const path = createPath(StudentState);
console.log("Ruta:", path); // Arreglo con la ruta a seguir
