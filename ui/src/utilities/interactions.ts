export function scrollToSection(id: string) {
  // jump if valid id has been provided
  if (document.getElementById(id)) {
    // document.getElementById(value).getBoundingClientRect().top);
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
  } else {
    console.log('Error: scroll target does not exist!');
  }
}
