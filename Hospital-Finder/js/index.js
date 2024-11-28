document.addEventListener("DOMContentLoaded", () => {
  let myForm = document.querySelector("form").addEventListener("submit", (e) => {
    e.preventDefault();
    document.querySelector('#map').scrollIntoView({
      behavior: 'smooth'
    });
    e.target.reset();

  })
});

mapboxgl.accessToken = 'pk.eyJ1IjoiZXN0aGVyY2F0ZXYiLCJhIjoiY2wyN2w3M256MDBqYjN0bW1uOG16ZzVqdiJ9.apozKCwK2RIwWPweckfjSg';

navigator.geolocation.getCurrentPosition(successLocation, errorLocation, {
  enableHighAccuracy: true
})

function successLocation(position) {
  console.log(position);
  setUpMap([position.coords.longitude, position.coords.latitude]);
}

function errorLocation() {
  setUpMap([36.817223, -1.286389])
}

function setUpMap(center) {
  const map = new mapboxgl.Map({
    container: 'map',
    style: 'mapbox://styles/esthercatev/cl29khs4v000514ms5eiyerq3',
    center: center,
    zoom: 15
  })

  //adds direction plugin
  map.addControl(
    new MapboxDirections({
      accessToken: mapboxgl.accessToken
    }),
    'top-left'
  );

  fetch('/nearby-hospitals')
  .then(response => response.json())
  .then(data => {
    const hospitalsContainer = document.getElementById('hospitals-container');
    hospitalsContainer.innerHTML = ''; // Clear existing content

    data.forEach(hospital => {
      const hospitalDiv = document.createElement('div');
      hospitalDiv.innerHTML =
        <h3>${hospital.name || "Name not available"}</h3>
        <p><strong>Specialty:</strong> ${hospital.specialty || "Not mentioned"}</p>
        <p><strong>Distance:</strong> ${hospital.distance || "Unknown"} km</p>
        <p><strong>Rating:</strong> ${hospital.rating || "Not rated"}</p>
        <hr>
      ;
      hospitalsContainer.appendChild(hospitalDiv);
    });
  })
  .catch(error => {
    console.error('Error fetching data:', error);
  });


  const marker = new mapboxgl.Marker({
      color: "#ff0000",
      draggable: true
    }).setLngLat([36.9011852, -1.318583])
    .addTo(map);

  //adds zoom and rotate controls
  const navControls = map.addControl(new mapboxgl.NavigationControl());
}