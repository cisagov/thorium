import React, { useEffect, useRef, useState } from 'react';
import { useParams } from 'react-router-dom';
import { ButtonToolbar, ButtonGroup, Button, Card } from 'react-bootstrap';
import { FaAngleDown, FaAngleDoubleDown, FaAngleDoubleUp, FaAngleUp } from 'react-icons/fa';

// project imports
import { OverlayTipBottom, Page } from '@components';
import { useAuth } from '@utilities';
import { getReactionStageLogs } from '@thorpi';

const ReactionStageLogs = () => {
  const { reactionID } = useParams();
  const { group } = useParams();
  const { stage } = useParams();
  const [cursor, setCursor] = useState(0);
  const [startLogLine, setStartLogLine] = useState(0);
  const [endLogLine, setEndLogLine] = useState(0);
  const [tailLogs, setTailLogs] = useState(true);
  const [pagingDown, setPagingDown] = useState(false);
  const [reactionStageLogs, setReactionStageLogs] = useState([]);
  const [logUpdateTimeout, setLogUpdateTimeout] = useState(10);
  const { checkCookie } = useAuth();
  const maxRenderedLines = 1000;
  const limit = 1000;

  const getLogPage = async (start) => {
    const stageLogs = await getReactionStageLogs(group, reactionID, stage, checkCookie, cursor, limit);

    if (!stageLogs) {
      setLogUpdateTimeout(10000);
      return;
    }

    // Add any new logs to the current log page
    if (stageLogs.length == 0) {
      setTailLogs(true);
      setLogUpdateTimeout(10000);
      return;
    } else {
      setLogUpdateTimeout(100);
    }

    let logs = [];
    // getting previous page
    if (startLogLine > cursor && startLogLine - cursor > endLogLine - startLogLine) {
      // get log lines when there is no overlap with the locally cached logs
      setStartLogLine(cursor);
      setEndLogLine(cursor + stageLogs.length);
      logs = stageLogs;
    } else if (startLogLine > cursor) {
      // get previous log page when there is overlap with the cached logs
      const endLogBufferIndex = reactionStageLogs.length - (startLogLine - cursor);
      logs = stageLogs.concat(reactionStageLogs.slice(0, endLogBufferIndex));
      setStartLogLine(cursor);
      setEndLogLine(cursor + logs.length);
    } else {
      // get the next page and append to cached log lines
      let logStart = startLogLine;
      if (endLogLine + stageLogs.length - startLogLine > maxRenderedLines) {
        logStart = endLogLine + stageLogs.length - maxRenderedLines;
      }
      let startLogBufferIndex = 0;
      if (logStart > 0) {
        startLogBufferIndex = stageLogs.length;
      }
      logs = reactionStageLogs.slice(startLogBufferIndex).concat(stageLogs);
      setStartLogLine(logStart);
      setEndLogLine(endLogLine + stageLogs.length);
    }
    // set the logs so that the page rerenders
    setReactionStageLogs(logs);
  };

  const pageToEndAndFollow = () => {
    setTailLogs(true);
    setPagingDown(true);
    setCursor(endLogLine);
  };

  const pageDown = () => {
    setTailLogs(false);
    setPagingDown(true);
    setCursor(endLogLine);
  };

  const pageToTop = () => {
    setTailLogs(false);
    setPagingDown(false);
    setCursor(0);
  };

  const pageUp = () => {
    setTailLogs(false);
    setPagingDown(false);
    if (startLogLine - limit < 0) {
      setCursor(0);
    } else {
      setCursor(startLogLine - limit);
    }
  };

  // get first page when we initially load the logs page
  useEffect(() => {
    getLogPage(startLogLine);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Get reaction info and logs every logUpdateTimeout seconds
  // When the end of the logs is reached this timeout will be increased to
  // reduce the number of API requests
  useEffect(() => {
    const intervalId = setInterval(() => {
      // only get a log update if tailing logs
      if (tailLogs) {
        // update cursor if requesting new page of logs
        if (endLogLine != cursor) {
          setCursor(endLogLine);
          // if page was previously requested and empty we need to
          // grab the page itself since the useEffect only calls on cursor
          // change and we already updated the cursor
        } else {
          getLogPage(cursor);
        }
      }
    }, logUpdateTimeout);
    return () => clearInterval(intervalId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tailLogs, endLogLine, logUpdateTimeout]);

  // When the cursor is updated by the interval or button actions,
  // get the requested page and trigger a rerender
  useEffect(() => {
    if (cursor < startLogLine || cursor >= endLogLine) {
      getLogPage(cursor);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cursor]);

  // Scroll to end when logs are updated and the update action is a scroll
  // down action otherwise scroll up
  const startLogsRef = useRef(null);
  const endLogsRef = useRef(null);
  useEffect(() => {
    const scrollDown = () => {
      endLogsRef.current?.scrollIntoView(false, {
        block: 'end',
        behavior: 'auto',
      });
    };
    const scrollUp = () => {
      startLogsRef.current?.scrollIntoView(true, {
        block: 'start',
        behavior: 'auto',
      });
    };
    if (pagingDown) {
      scrollDown();
    } else {
      scrollUp();
    }
  }, [reactionStageLogs, pagingDown]);

  return (
    <Page title="Reaction Stage Logs">
      <br />
      <Card className="log-box panel">
        <Card.Header>
          <ButtonToolbar className="d-flex justify-content-center">
            <ButtonGroup>
              <Button variant="" onClick={pageToTop} className="log-nav-button primary-btn">
                <OverlayTipBottom tip={'Scroll to start of logs'}>
                  <FaAngleDoubleUp size={18} />
                </OverlayTipBottom>
              </Button>
              <Button variant="" onClick={pageUp} className="log-nav-button primary-btn">
                <OverlayTipBottom tip={'Scroll up in logs'}>
                  <FaAngleUp size={18} />
                </OverlayTipBottom>
              </Button>
              <Button variant="" onClick={pageDown} className="log-nav-button primary-btn">
                <OverlayTipBottom tip={'Scroll down in logs'}>
                  <FaAngleDown size={18} />
                </OverlayTipBottom>
              </Button>
              <Button variant="" onClick={pageToEndAndFollow} className="log-nav-button primary-btn">
                <OverlayTipBottom tip={'Scroll to end of logs and follow'}>
                  <FaAngleDoubleDown size={18} />
                </OverlayTipBottom>
              </Button>
            </ButtonGroup>
          </ButtonToolbar>
        </Card.Header>
        <Card.Body className="scrollable-card">
          {reactionStageLogs &&
            reactionStageLogs.map((line, idx) => (
              <div key={startLogLine + idx} className="raw-log-line" ref={endLogsRef}>
                <i className="log-line-index secondary-text">{startLogLine + idx}&emsp;</i>
                {line}
              </div>
            ))}
        </Card.Body>
      </Card>
    </Page>
  );
};

export default ReactionStageLogs;
