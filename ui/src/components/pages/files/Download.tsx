import { useState } from 'react';
import { Col, Form, Row } from 'react-bootstrap';
import AlertBanner, { Severity } from '@components/shared/alerts/AlertBanner';
import { FaDownload } from 'react-icons/fa';

// project imports
import { downloadBlob } from '@utilities/download';
import { getFile } from '@thorpi/files';

// spec: ./files.spec.md

type ArchiveFormat = 'CaRT' | 'Encrypted ZIP';
const Formats: ArchiveFormat[] = ['CaRT', 'Encrypted ZIP'];

// fetch the archived sample and trigger a browser download
const downloadFile = async (
  sha256: string,
  setDownloadFileError: (e: string) => void,
  archiveFormat: ArchiveFormat,
  archivePassword: string,
) => {
  const res = await getFile(sha256, setDownloadFileError, archiveFormat, archivePassword);
  if (res) {
    downloadBlob(res, `${sha256}.${archiveFormat == 'CaRT' ? 'cart' : 'zip'}`);
  }
};

interface DownloadProps {
  sha256: string;
}

const Download = ({ sha256 }: DownloadProps) => {
  const [downloadFileError, setDownloadFileError] = useState('');
  const [archiveFormat, setArchiveFormat] = useState<ArchiveFormat>('Encrypted ZIP');
  const [archivePassword, setArchivePassword] = useState('');
  return (
    <div className="mt-4" id="download-tab">
      {downloadFileError && (
        <Row>
          <Col>
            <AlertBanner severity={Severity.Warning}>{downloadFileError}</AlertBanner>
          </Col>
        </Row>
      )}
      <Form>
        <Row className="d-flex justify-content-center">
          <Col className="d-flex justify-content-end mt-3">Format</Col>
          <Col className="d-flex justify-content-start">
            <Form.Group controlId="downloadForm.FormatSelector">
              <Form.Select value={archiveFormat} onChange={(e) => setArchiveFormat(e.target.value as ArchiveFormat)}>
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
            onClick={() => {
              void downloadFile(sha256, setDownloadFileError, archiveFormat, archivePassword == '' ? 'infected' : archivePassword);
            }}
          >
            <FaDownload size="120" />
          </a>
        </Col>
      </Row>
    </div>
  );
};

export default Download;
