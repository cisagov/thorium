import { useEffect, useState } from 'react';
import { Alert, Table } from 'react-bootstrap';
import styled from 'styled-components';

// project imports
import Page from '@components/pages/Page';
import Subtitle from '@components/shared/titles/Subtitle';
import Title from '@components/shared/titles/Title';
import { getSystemSettings } from '@thorpi/system';

// Styled components
const Settings = styled.div`
  display: flex;
  align-items: center;
  flex-direction: column;
  padding-top: 10px;
`;

const SystemSettings = () => {
  const [getSettingsError, setGetSettingsError] = useState('');
  const [systemSettings, setSystemSettings] = useState({});

  // fetch latest stats from API
  const fetchSettings = async () => {
    const settings = await getSystemSettings(setGetSettingsError);
    if (settings) {
      setSystemSettings(settings);
    }
  };

  // trigger fetch stats on initial page load
  useEffect(() => {
    fetchSettings();
  }, []);

  return (
    <Page title="Settings · Thorium" className="settings">
      <Settings>
        {getSettingsError != '' && <Alert>{getSettingsError}</Alert>}
        <Title>System Settings</Title>
        <Table striped bordered hover>
          <tbody>
            {Object.keys(systemSettings).map((setting) => (
              <tr key={setting}>
                <td>
                  <center>
                    <Subtitle>{setting}</Subtitle>
                  </center>
                </td>
                <td>
                  <center>
                    <Subtitle>{systemSettings[setting].toString()}</Subtitle>
                  </center>
                </td>
              </tr>
            ))}
          </tbody>
        </Table>
      </Settings>
    </Page>
  );
};

export default SystemSettings;
