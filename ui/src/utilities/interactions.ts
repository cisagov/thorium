/**
 * Smoothly scroll the element with the given id into view.
 *
 * If no element with that id exists, logs an error and does nothing.
 *
 * @param id - The DOM id of the element to scroll to.
 */
export function scrollToSection(id: string) {
  // jump if valid id has been provided
  if (document.getElementById(id)) {
    // document.getElementById(value).getBoundingClientRect().top);
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
  } else {
    console.log('Error: scroll target does not exist!');
  }
}
