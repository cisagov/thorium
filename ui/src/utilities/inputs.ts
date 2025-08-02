import { parseISO } from 'date-fns';

// safely convert formatted string to date
export function safeStringToDateConversion(date: string | null) {
  try {
    return parseISO(date as unknown as string);
  } catch (e) {
    if (e instanceof RangeError) {
      // this hits if the passed in vaqlue is not a valid date
      return null;
    } else {
      throw e;
    }
  }
}

// safely convert date object to string
export function safeDateToStringConversion(date: Date | null) {
  if (date == null) {
    return null;
  }
  try {
    return date.toISOString();
  } catch (e) {
    if (e instanceof RangeError) {
      // this hits if the passed in value is not a valid date
      return null;
    } else {
      throw e;
    }
  }
}

// safely parse JSON from things like session or input fields
export function safeParseJSON(unsafeJSON: string) {
  try {
    return JSON.parse(unsafeJSON);
  } catch (e) {
    if (e instanceof SyntaxError) {
      // this hits if the passed in value is not a valid date
      return null;
    } else {
      throw e;
    }
  }
}
