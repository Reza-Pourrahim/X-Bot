import axios from "axios";

// function prepareRequestParameters(obj) {
//   const params = new URLSearchParams();
//   Object.keys(obj).forEach((k) => {
//     params.append(k, obj[k]);
//   });
//
//   return params;
// }

const instance = axios.create({
  baseURL: "http://127.0.0.1:8000",
});

function getSingleEndpoint(parameters, endpoint) {
  // const params = {
  //   headers: {
  //     Authorization: 'Bearer {{my token here}}',
  //   }
  // }

  const options = {
    ...parameters,
  };

  const args = Object.entries(options)
    .map((d) => `${d[0]}=${d[1]}`)
    .join("&");

  return instance.get(`/${endpoint}?${args}`);
}

export { instance, getSingleEndpoint };
