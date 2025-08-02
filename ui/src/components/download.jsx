import React, { useState } from 'react';
import { Alert, Col, Form, Row } from 'react-bootstrap';
import { FaDownload } from 'react-icons/fa';

// project imports
import { getFile } from '@thorpi';

const Formats = ['CaRT', 'Encrypted ZIP'];

const downloadFile = async (sha256, setDownloadFileError, archiveFormat, archivePassword) => {
  const res = await getFile(sha256, setDownloadFileError, archiveFormat, archivePassword);
  if (res) {
    // turn response data to blob object
    const blob = new Blob([res]);
    // map url to blob in memory
    const url = window.URL.createObjectURL(blob);
    // create anchor tag for blob link
    const link = document.createElement('a');
    // assign href
    link.href = url;
    // set link as download
    link.setAttribute('download', `${sha256}.${archiveFormat == 'CaRT' ? 'cart' : 'zip'}`);
    // Append to html link element page
    document.body.appendChild(link);
    // Start download
    link.click();
    // Clean up and remove the link
    link.parentNode.removeChild(link);
  }
};

export const Download = ({ sha256 }) => {
  const [downloadFileError, setDownloadFileError] = useState('');
  const [archiveFormat, setArchiveFormat] = useState('Encrypted ZIP');
  const [archivePassword, setArchivePassword] = useState('');

  return (
    <div className="mt-4" id="download-tab">
      {downloadFileError && (
        <Row>
          <Col>
            <Alert variant="warning" className="d-flex justify-content-center">
              {downloadFileError}
            </Alert>
          </Col>
        </Row>
      )}
      <Form>
        <Row className="d-flex justify-content-center">
          <Col className="d-flex justify-content-end mt-3">Format</Col>
          <Col className="d-flex justify-content-start">
            <Form.Group controlId="downloadForm.FormatSelector">
              <Form.Select value={archiveFormat} onChange={(e) => setArchiveFormat(String(e.target.value))}>
                {Formats.map((format) => (
                  <option key={format} value={format}>
                    {format}
                  </option>
                ))}
              </Form.Select>
            </Form.Group>
          </Col>
        </Row>
        {archiveFormat == 'Encrypted ZIP' && (
          <Row className="d-flex justify-content-center mt-3">
            <Col className="d-flex justify-content-end mt-3">
              <span>Password</span>
            </Col>
            <Col className="d-flex justify-content-start">
              <Form.Group controlId="downloadForm.PasswordInput">
                <Form.Control
                  type="password"
                  value={archivePassword}
                  placeholder="infected"
                  onChange={(e) => setArchivePassword(String(e.target.value))}
                />
              </Form.Group>
            </Col>
          </Row>
        )}
      </Form>
      <Row>
        <Col className="d-flex justify-content-center mt-5">
          <a
            className="d-flex justify-content-center download-btn"
            href="#download"
            onClick={() => downloadFile(sha256, setDownloadFileError, archiveFormat, archivePassword == '' ? 'infected' : archivePassword)}
          >
            <FaDownload size="120" />
          </a>
        </Col>
      </Row>
    </div>
  );
};
