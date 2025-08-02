import React, { useState, useEffect } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { Alert, Button, Col, Card, Form, Modal, Row } from 'react-bootstrap';

// project imports
import { LoadingSpinner, SimpleTitle, SimpleSubtitle, Subtitle, Title, Page } from '@components';
import { useAuth } from '@utilities';
import { getBanner } from '@thorpi';

const LoginContainer = () => {
  const [showRegModal, setShowRegModal] = useState(false);
  const handleCloseRegModal = () => setShowRegModal(false);
  const handleShowRegModal = () => setShowRegModal(true);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loginErr, setLoginErr] = useState('');
  const [banner, setBanner] = useState('');
  const [loggingIn, setLoggingIn] = useState(false);
  const navigate = useNavigate();
  const { state } = useLocation();
  const { login } = useAuth();

  // login to Thorium and redirect if successful
  const handleAuthFormSubmit = async (username, password, handleAuthErr) => {
    setLoggingIn(true);
    setLoginErr('');
    if (await login(username, password, handleAuthErr)) {
      navigate(state?.path || '/');
    } else {
      setLoggingIn(false);
    }
  };

  // handle all key presses
  const handleFormKeyPress = async (e) => {
    // key code 13 is enter
    if (e.keyCode === 13) {
      handleAuthFormSubmit(username, password, setLoginErr);
    }
  };

  // fetch banner and set state
  const fetchBanner = async () => {
    const req = await getBanner(setBanner);
    if (req) {
      setBanner(req);
    }
  };

  // async grab banner on page load
  useEffect(() => {
    fetchBanner();
  }, []);

  const RegisterModal = () => {
    const localAuth = false;
    const [regUsername, setRegUsername] = useState('');
    const [regEmail, setRegEmail] = useState('');
    const [regPass, setRegPass] = useState('');
    const [regVerifyPass, setRegVerifyPass] = useState('');
    const [regError, setRegError] = useState('');
    const [regWarning, setRegWarning] = useState('');
    const [registering, setRegistering] = useState(false);
    const { register } = useAuth();

    const handleRegister = async (username, email, pass, verifyPass) => {
      // handleRegister is a form submit, we want to clear the warnings
      // related to form entry issues
      setRegWarning('');
      setRegError('');
      setRegistering(true);
      // make sure all required fields are present
      if (username && pass && (verifyPass || !localAuth)) {
        if (localAuth && pass != verifyPass) {
          setRegWarning('The entered passwords do not match');
          setRegistering(false);
        } else {
          // create the user account
          register(username, pass, setRegError, email, 'User').then((res) => {
            // redirect after successful registration of user
            if (res) {
              navigate(state?.path || '/');
            } else {
              setRegistering(false);
            }
          });
        }
      } else {
        setRegWarning('You must specify a username and password!');
        setRegistering(false);
      }
      // registration failed, display error and do not redirect
      return;
    };

    // handle all key presses
    const checkEnterSubmit = async (e) => {
      // key code 13 is enter
      if (e.keyCode === 13) {
        handleRegister(regUsername, regEmail, regPass, regVerifyPass);
      }
    };

    return (
      <Modal show={showRegModal} onHide={handleCloseRegModal}>
        <Modal.Header closeButton>
          <Modal.Title>
            <Title>Register</Title>
          </Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <Form>
            <Row>
              <Col>
                <Form.Group>
                  <Form.Label>
                    <Subtitle>Username</Subtitle>
                  </Form.Label>
                  <Form.Control
                    type="text"
                    placeholder="Enter Username"
                    value={regUsername}
                    onKeyDown={(e) => checkEnterSubmit(e)}
                    onChange={(e) => setRegUsername(String(e.target.value))}
                  />
                </Form.Group>
              </Col>
            </Row>
            <Row>
              <Col>
                <Form.Group>
                  <Form.Label>
                    <Subtitle>Email</Subtitle>
                  </Form.Label>
                  <Form.Control
                    type="text"
                    placeholder="Enter Email"
                    value={regEmail}
                    onKeyDown={(e) => checkEnterSubmit(e)}
                    onChange={(e) => setRegEmail(String(e.target.value))}
                  />
                </Form.Group>
              </Col>
            </Row>
            <Row className="my-3">
              <Col>
                <Form.Group>
                  <Form.Label>
                    <Subtitle>Password</Subtitle>
                  </Form.Label>
                  <Form.Control
                    type="password"
                    placeholder="Enter Password"
                    value={regPass}
                    onKeyDown={(e) => checkEnterSubmit(e)}
                    onChange={(e) => setRegPass(String(e.target.value))}
                  />
                  {localAuth && (
                    <>
                      <Form.Label>
                        <Subtitle>Verify Password</Subtitle>
                      </Form.Label>
                      <Form.Control
                        type="password"
                        placeholder="Verify Password"
                        value={regVerifyPass}
                        onKeyDown={(e) => checkEnterSubmit(e)}
                        onChange={(e) => setRegVerifyPass(String(e.target.value))}
                      />
                    </>
                  )}
                </Form.Group>
              </Col>
            </Row>
            {regWarning != '' && (
              <Row>
                <Alert variant="warning">
                  <center>{regWarning}</center>
                </Alert>
              </Row>
            )}
            {regError != '' && (
              <Row>
                <Alert variant="danger">
                  <center>{regError}</center>
                </Alert>
              </Row>
            )}
            {registering ? (
              <LoadingSpinner loading={registering}></LoadingSpinner>
            ) : (
              <Row>
                <Col className="d-flex justify-content-center">
                  <Button className="ok-btn m-2" onClick={() => handleRegister(regUsername, regEmail, regPass, regVerifyPass)}>
                    Submit
                  </Button>
                </Col>
              </Row>
            )}
          </Form>
        </Modal.Body>
      </Modal>
    );
  };

  return (
    <Page title="Login · Thorium">
      <Row>
        <Col className="d-flex justify-content-center align-items-center">
          <Card className="p-4 d-flex justify-content-center align-items-center panel">
            <Card.Title>
              <SimpleTitle>Welcome to Thorium!</SimpleTitle>
            </Card.Title>
            <Card.Body>
              {banner != null && banner != '' && (
                <Row>
                  <Col className="d-flex justify-content-center">
                    <pre className="banner">
                      <center>{String(banner)}</center>
                    </pre>
                  </Col>
                </Row>
              )}
              <Row>
                <Col className="d-flex justify-content-center">
                  <Form.Control
                    className="m-2 login"
                    type="text"
                    value={username}
                    placeholder="username"
                    onChange={(e) => setUsername(String(e.target.value))}
                    onKeyDown={(e) => handleFormKeyPress(e)}
                  />
                </Col>
              </Row>
              <Row>
                <Col className="d-flex justify-content-center">
                  <Form.Control
                    className="m-2 login"
                    type="password"
                    value={password}
                    placeholder="password"
                    onChange={(e) => setPassword(String(e.target.value))}
                    onKeyDown={(e) => handleFormKeyPress(e)}
                  />
                </Col>
              </Row>
              {loggingIn ? (
                <LoadingSpinner loading={loggingIn}></LoadingSpinner>
              ) : (
                <>
                  <Row className="mt-3">
                    <Col className="d-flex justify-content-center align-items-center">
                      <SimpleSubtitle>
                        New user? Create an&nbsp;
                        <Link to="/auth" onClick={() => handleShowRegModal()}>
                          account
                        </Link>
                        .
                      </SimpleSubtitle>
                    </Col>
                  </Row>
                  <Row>
                    {loginErr != '' && (
                      <center>
                        <Alert variant="danger">{loginErr}</Alert>
                      </center>
                    )}
                  </Row>
                  <Row>
                    <Col className="d-flex justify-content-center">
                      <Button
                        className="primary-btn"
                        onClick={() => handleAuthFormSubmit(username, password, setLoginErr)}
                        variant="success"
                      >
                        Login
                      </Button>
                    </Col>
                  </Row>
                </>
              )}
            </Card.Body>
            <RegisterModal />
          </Card>
        </Col>
      </Row>
    </Page>
  );
};

export default LoginContainer;
