import React, { useEffect, useState } from 'react';
import { Alert } from 'react-bootstrap';

// project imports
const ToolResult = React.lazy(() => import('@components/tools/ToolResult'));
import LoadingSpinner from '@components/shared/fallback/LoadingSpinner';
import { useAuth } from '@utilities/auth';
import { updateURLSection } from '@utilities/url';
import { scrollToSection } from '@utilities/interactions';
import { getResults } from '@thorpi/results';

// floating table of contents object
const ResultsTableOfContents = ({ parsedResults, inViewElements }) => {
  return (
    <nav className="results-toc">
      <ul className="ul no-bullets">
        {parsedResults &&
          typeof parsedResults === 'object' &&
          Object.keys(parsedResults)
            .sort()
            .map((image) => (
              <li key={`results-${image}-toc`} className="results-toc-item">
                <a
                  href={`#results-${image}`}
                  onClick={() => scrollToSection(`results-tab-${image}`)}
                  className={`${inViewElements.includes(image) ? 'selected' : 'unselected'}`}
                >
                  {image}
                </a>
                <hr className="m-1" />
              </li>
            ))}
      </ul>
    </nav>
  );
};

const Results = ({ sha256, results, setResults, numResults, setNumResults }) => {
  const parsedResults = structuredClone(results);
  // whether content is currently loading
  const [loading, setLoading] = useState(false);
  const [inViewElements, setInViewElements] = useState([]);
  const { checkCookie } = useAuth();
  // get results from API
  useEffect(() => {
    let isSubscribed = true;
    const fetchData = async () => {
      setLoading(true);
      const resultsRes = await getResults(sha256, checkCookie, {});
      // results must be set and
      // there can only be one outstanding subscribed request
      if (resultsRes && 'results' in resultsRes && isSubscribed) {
        // pass back number of results to parent
        setNumResults(Object.keys(resultsRes.results).length);
        setResults(resultsRes.results);
      }
      setLoading(false);
    };
    fetchData();
    return () => {
      isSubscribed = false;
    };
  }, [sha256]);

  // update whether an element is in the view port
  const updateInView = (inView, entry) => {
    if (inView) {
      setInViewElements((previousInViewElements) => [...previousInViewElements, entry].sort());
    } else {
      setInViewElements((previousInViewElements) => {
        return previousInViewElements.filter((element) => element != entry).sort();
      });
    }
  };

  // remove hidden display typed results from results object
  Object.keys(parsedResults)
    .sort()
    .map((image) => {
      if (parsedResults[image][0]['display_type'] && results[image][0]['display_type'] == 'Hidden') {
        delete parsedResults[image];
      }
    });

  return (
    <div id="results-tab" className="navbar-scroll-offset results-container">
      <LoadingSpinner loading={loading}></LoadingSpinner>
      {parsedResults && typeof parsedResults === 'object' && !loading && (
        <>
          <div>
            {numResults == 0 && !loading && (
              <>
                <br />
                <Alert variant="" className="info">
                  <Alert.Heading>
                    <center>
                      <h3>No Tool Results Available</h3>
                    </center>
                  </Alert.Heading>
                  <center>
                    <p>Check back later for updated results</p>
                  </center>
                </Alert>
              </>
            )}
            {Object.keys(parsedResults)
              .sort()
              .map((image) => (
                <ToolResult
                  key={image}
                  header={image}
                  type={parsedResults[image][0]['display_type'] ? parsedResults[image][0]['display_type'] : 'Json'}
                  tool={image}
                  sha256={sha256}
                  updateInView={updateInView}
                  updateURLSection={updateURLSection}
                  result={parsedResults[image][0]}
                />
              ))}
          </div>
          {Object.keys(parsedResults).length > 0 && (
            <div className="results-toc-col">
              <ResultsTableOfContents parsedResults={parsedResults} inViewElements={inViewElements} />
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default Results;
