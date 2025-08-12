import axios from 'axios';
import JSONbig from 'json-bigint';
// Import GUI config file
const CONFIG = import.meta.glob('./config.json') as any;

/**
 * Get token from the cookie
 * @function
 * @param {string} cname - name of cookie to get.
 * @returns {string}- token or empty if blank.
 */
function getCookie(cname: string): string {
  const name = cname + '=';
  const decodedCookie = decodeURIComponent(document.cookie);
  const ca = decodedCookie.split(';');
  for (let i = 0; i < ca.length; i++) {
    let c = ca[i];
    while (c.charAt(0) == ' ') {
      c = c.substring(1);
    }
    if (c.indexOf(name) == 0) {
      return c.substring(name.length, c.length);
    }
  }
  return '';
}

// API should either be set with REACT_APP_API_URL or defined by the URL that
// was used to navigate to the frontend
let apiURL = '';
if (window.location.hostname == 'localhost' || window.location.hostname == '127.0.0.1') {
  // Used in when running local dev instance and pointed to a remote API
  if (process.env.REACT_APP_API_URL && process.env.REACT_APP_API_URL !== '') {
    apiURL = `${process.env.REACT_APP_API_URL}`;
  } else {
    apiURL = `${window.location.protocol}//${window.location.hostname}/api`;
  }
} else {
  apiURL = `${window.location.protocol}//${window.location.hostname}/api`;
}
// Create axios instance using config file opts
const client = axios.create({
  baseURL: apiURL,
});

// Inject token as a parameter to support legacy silo routes
client.interceptors.request.use((config) => {
  for (const header in CONFIG.headers) {
    if (CONFIG.headers.hasOwnProperty(header)) {
      config.headers[header] = CONFIG.headers[header];
    }
  }
  // Make sure header is set if not passed into client request
  if (typeof config.headers.Authorization === 'undefined') {
    config.headers.Authorization = 'token ' + btoa(getCookie('THORIUM_TOKEN'));
  }
  return config;
});

// Create axios instance using config file opts
const bigIntClient = axios.create({
  baseURL: apiURL,
  transformResponse: [
    function (data) {
      try {
        return JSONbig.parse(data);
      } catch (error) {
        return data;
      }
    },
  ],
});

// Inject token as a parameter to support legacy silo routes
bigIntClient.interceptors.request.use((config) => {
  for (const header in CONFIG.headers) {
    if (CONFIG.headers.hasOwnProperty(header)) {
      config.headers[header] = CONFIG.headers[header];
    }
  }
  // Make sure header is set if not passed into client request
  if (typeof config.headers.Authorization === 'undefined') {
    config.headers.Authorization = 'token ' + btoa(getCookie('THORIUM_TOKEN'));
  }
  return config;
});

function parseRequestError(error: string, errorHandler: (error: string) => void, requestType: string) {
  if (axios.isAxiosError(error)) {
    if (error.response) {
      const trace = error.response.data.trace ? `trace: ${error.response.data.trace}` : '';
      const errorStatus = error.response.status == 401 ? 'Permission Denied' : error.response.status;
      const errorMsg = error.response.data.error ? error.response.data.error : errorStatus;
      errorHandler(`Failed to ${requestType}: ${errorMsg} ${trace}`);
    } else if (error.request) {
      errorHandler(`Failed to receive a ${requestType} request response: "${error.request}`);
    } else {
      errorHandler(`Failed to setup ${requestType} request: "${error.message}"`);
    }
  } else {
    errorHandler(`Unexpected error: ${error}`);
  }
}

export { client as default, bigIntClient, parseRequestError };
