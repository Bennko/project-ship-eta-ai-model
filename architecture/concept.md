## User Interface - Web based server-client architecture

For our UI we wanted to imlement a classic web based server-client architecture with the following components:
- Frontend: HTML, CSS (Bootstrap), Javascript
    - Since our needs are simple using this instead of frameworks makes things easier
- Backend: Node.js (Express)
    - Simple to set up and not much overhead like Django for example where there are a lot of already built in functionalities which we don't need, that make the language complex to learn and apply
- Ett service (Flask)
    - Use joblib to load the model into the service
    - Ett service separate in a docker container for encapsulation and scalibility
    - Flask like Express is also the bare minimun of functinality and therefore suites our simple needs

All of this is run in 2 separate docker container that are started up simutaneously and put on a networkfor communication by docker compose as specified in the "docker-compose.yml" file
- This makes deployment in any kind of environment simple as long as docker and docker compose are available



