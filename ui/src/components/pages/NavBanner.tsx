import { Link, useNavigate } from 'react-router-dom';
import { Nav, Navbar, NavDropdown } from 'react-bootstrap';
import { FaQuestion } from 'react-icons/fa';

// project imports
import { OverlayTipRight } from '@components/shared/overlay/tips';
import { clearTagDataFromLocalStorage } from '@utilities/tags';
import { useAuth } from '@utilities/auth';
import { getApiUrl } from '@utilities/url';

const NavBanner = () => {
  const { userInfo, logout } = useAuth();
  const navigate = useNavigate();
  const apiURL = getApiUrl();

  // call auth logout and redirect to login
  const handleLogout = () => {
    clearTagDataFromLocalStorage();
    // re-render navbar to remove username
    logout().then(() => {
      navigate('/');
      // force reload of site on logout
      window.location.reload();
    });
  };

  // go to the root
  const handleHomeClick = () => {
    if (window.location.pathname.startsWith('/?') || window.location.pathname == '/') {
      // if we're already at the root, force a page reload to reload the search page
      window.location.href = '/';
    } else {
      // otherwise just navigate there
      navigate('/');
    }
  };

  return (
    <Navbar className="navbar-banner panel d-flex justify-content-end">
      <Nav.Link className="home-item" onClick={handleHomeClick}>
        <OverlayTipRight tip={'Home'}>
          <img src="/ferris-scientist.png" alt="FerrisScientist" width="40px" />
        </OverlayTipRight>
      </Nav.Link>
      <Nav className="d-flex justify-content-end">
        {
          <Navbar.Brand className="navbanner-item mx-2 px-2 pb-3" href={`${apiURL}/docs/user/index.html`}>
            <FaQuestion className="mt-3" size={22} />
          </Navbar.Brand>
        }
        {userInfo && userInfo.username && (
          <NavDropdown align="end" className="navbanner-item" title={`@${userInfo.username}`}>
            <NavDropdown.Item as={Link} to="/profile">
              Profile
            </NavDropdown.Item>
            <NavDropdown.Item onClick={() => handleLogout()}>Logout</NavDropdown.Item>
          </NavDropdown>
        )}
      </Nav>
    </Navbar>
  );
};

export default NavBanner;
