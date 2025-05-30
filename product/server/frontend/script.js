function getDifferenceSeconds(date1, date2){
    var parsedDate1 = new Date(date1)
    var parsedDate2 = new Date(date2);
    var timeDiff = Math.abs(parsedDate2.getTime() - parsedDate1.getTime()); // in miliseconds
    var timeDiffInSecond = Math.ceil(timeDiff / 1000); // in second
    return timeDiffInSecond;
}

function setTimeToClock(ettSec){
    var days = Math.floor(ettSec / (3600*24));
    ettSec  -= days*3600*24;
    var hrs   = Math.floor(ettSec / 3600);
    ettSec  -= hrs*3600;
    var mnts = Math.floor(ettSec / 60);
    ettSec  -= mnts*60;
    document.getElementById("days").innerHTML = days.
        toString().padStart(2, '0');
    document.getElementById("hours").innerHTML = hrs.
        toString().padStart(2, '0');
    document.getElementById("minutes").innerHTML = mnts.
        toString().padStart(2, '0');
    document.getElementById("seconds").innerHTML = ettSec.
        toString().padStart(2, '0');
}

async function fetchEtt(latitude, longitude, sog, th, cog, shiptype, endLatitude, endLongitude, pastTravelTime, destination){
    let response = await fetch("http://localhost:3000/ett", { //add error handling
        method: "POST",
        headers: {
            "Content-type": "application/json; charset=UTF-8"
        },
        body: JSON.stringify({
            latitude: latitude,
            longitude: longitude,
            sog: sog,
            th: th,
            cog: cog,
            shiptype: shiptype,
            endLatitude: endLatitude, 
            endLongitude: endLongitude, 
            pastTravelTime: pastTravelTime,
            destination: destination
        })
    });
    const data = response.json();
    return data;
}

document.addEventListener("DOMContentLoaded", function () {
    var map = L.map('map').setView([0, 0], 2);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(map);

    var marker;
    let pastLocations = [];

    document.getElementById('generate').addEventListener('click', async function () {
        // Get user data from form 
        var latitude = document.getElementById('latitude').value;
        var longitude = document.getElementById('longitude').value;
        var sog = document.getElementById('sog').value;
        var th = document.getElementById('th').value;
        var cog = document.getElementById('cog').value;
        var shiptype = document.getElementById('shiptype').value;
        var departureTime = document.getElementById('departure-time').value;
        var recordingTime = document.getElementById('recording-time').value;
        var destination = document.getElementById('destination').value;
        var endLatitude = document.getElementById('end-latitude').value;
        var endLongitude = document.getElementById('end-longitude').value;

        var pastTravelTime = getDifferenceSeconds(departureTime, recordingTime);
         
        //Get ett for data and set the UI clock
        let response = await fetchEtt(latitude, longitude, sog, th, cog, shiptype, endLatitude, endLongitude, pastTravelTime, destination);
        setTimeToClock(response.ett);

        const lastEntry = pastLocations[pastLocations.length - 1];

        // Delete previous marker and add it as point to map
        if(pastLocations.length){
            map.removeLayer(marker);
            marker = L.circle([lastEntry.latitude, lastEntry.longitude], {
                color: 'blue',
                radius: 5
            }).addTo(map).bindPopup(`Latitude: ${lastEntry.latitude}, <br>Longitude: ${lastEntry.longitude}, <br>SOG: ${lastEntry.sog}, <br>TH: ${lastEntry.th}, <br>COG: ${lastEntry.cog}, <br>Shiptype:${lastEntry.shiptype}, <br>Destination: ${lastEntry.destination}`);
        }

        // Create new marker
        marker = L.marker([latitude, longitude]).addTo(map)
            .bindPopup('Latitude: ' + latitude + '<br>Longitude: ' + longitude)
            .openPopup();
        map.setView([latitude, longitude], 10);

        // Add new location to past locations
        pastLocations.push({ latitude, longitude, sog, th, cog, shiptype, destination });
    });
});

document.addEventListener('DOMContentLoaded', (event) => {
    const destination = document.getElementById('destination');
    destination.addEventListener('change', () => {
        if(destination.value == "hamburg"){
            document.getElementById('end-latitude').value = "53.5";
            document.getElementById('end-longitude').value = "9.93";
        }else{
            document.getElementById('end-latitude').value = "51.97";
            document.getElementById('end-longitude').value = "4.02";
        }
    });
});