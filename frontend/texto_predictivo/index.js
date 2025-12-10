// Función que inicializa el evento de entrada de texto para predecirlo
function inicializar() {
  // Añade un evento de 'input' al campo de texto con id 'texto'
  document.getElementById('texto').addEventListener('input', function() {
    clearTimeout(timeout); // Limpia el temporizador previo si está activo

    // Establece un nuevo temporizador que realiza la traducción después de 1 segundo
    timeout = setTimeout(() => {
      cantidad = document.getElementById('cantidad').value;
      predecir(this.value, cantidad); // Llama a la función para predecir el texto introducido
    }, 1000); // Espera 1 segundo tras la última pulsación de tecla
  });

    // Añade un evento de 'input' al campo numérico con id 'cantidad'
    document.getElementById('cantidad').addEventListener('input', function() {
      clearTimeout(timeout); // Limpia el temporizador previo si está activo
  
      // Establece un nuevo temporizador que realiza la traducción después de 1 segundo
      timeout = setTimeout(() => {
        texto = document.getElementById('texto').value;
        predecir(texto, this.value); // Llama a la función para predecir el texto introducido
      }, 1000); // Espera 1 segundo tras la última pulsación de tecla
    });
}

// Función asíncrona que envía el texto al servidor para predecirlo y muestra el resultado
async function predecir(texto, cantidad) {
  const resultado = document.getElementById('resultado'); // Referencia al elemento donde se mostrará la predicción

  // Si el campo de texto está vacío, limpia el resultado y detiene la ejecución de la función
  if (texto.trim() === "") {
    resultado.innerHTML = "";
    return;
  }

  // Muestra un spinner mientras se espera la traducción
  resultado.innerHTML = `
    <div class="spinner-border text-primary" role="status"></div>
  `;

  try {
    // Realiza una solicitud fetch al servidor Python para obtener la traducción del texto
    let respuesta = await fetch(`${URL_PYTHON}/texto_predictivo?texto=${encodeURIComponent(texto)}&cantidad=${cantidad}`);
    respuesta = await respuesta.json(); // Convierte la respuesta en formato JSON

    // Muestra la traducción en el div resultado
    resultado.innerHTML = `<p>${respuesta.prediccion}...</p>`;
  } catch (error) {
    // Muestra un mensaje de error si la solicitud falla
    console.error("Error al predecir el texto:", error);
    resultado.innerHTML = "<span class='text-warning'>Error al predecir el texto</span>";
  }
}

// Inicializar los eventos y funcionalidades cuando se carga el script
inicializar();
