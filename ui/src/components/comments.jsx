import React, { Fragment, useEffect, useState } from 'react';
import { Alert, Button, Card, Col, Form, Pagination, Row } from 'react-bootstrap';

// project imports
import { UploadDropzone } from '@components/shared/uploaddropzone';
import { downloadAttachment, getFileDetails, postFileComments } from '@thorpi';

export const Comments = ({ sha256 }) => {
  const [newComment, setNewComment] = useState('');
  const [filesArray, setFilesArray] = useState([]);
  const [comments, setComments] = useState([]);
  const [limit, setLimit] = useState(0);
  const [maxPage, setMaxPage] = useState(100);
  const [page, setPage] = useState(0);
  const [commentError, setCommentError] = useState('');
  const PAGELIMIT = 10;

  const fetchComments = async () => {
    const fileDetails = await getFileDetails(sha256, setCommentError);

    if (fileDetails && fileDetails.comments) {
      setComments(fileDetails.comments);
      setMaxPage(Math.ceil(fileDetails.comments.length / PAGELIMIT));
      setLimit(PAGELIMIT);
    }
  };

  useEffect(() => {
    fetchComments();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sha256]);

  const getAttachment = async (commentID, name, fileID) => {
    const attachRes = await downloadAttachment(sha256, commentID, fileID);
    if (attachRes && attachRes.data && attachRes.headers) {
      // turn response data to blob object
      const blob = new Blob([attachRes.data], {
        type: attachRes.headers['content-type'],
      });
      // map url to blob in memory
      const url = window.URL.createObjectURL(blob);
      // create anchor tag for blob link
      const link = document.createElement('a');
      // assign href
      link.href = url;
      // set link as download
      link.setAttribute('download', name);
      // Append to html link element page
      document.body.appendChild(link);
      // Start download
      link.click();
      // Clean up and remove the link
      link.parentNode.removeChild(link);
    }
  };

  // handle event of posting to backend
  const handlePost = async (commentValue, filesArray) => {
    const form = new FormData();
    // check for possible space inserted comment
    if (commentValue.trim() === '') {
      setCommentError('Comment is Empty');
      return;
    } else {
      // add comment text to form
      form.append('comment', commentValue);
      // add file attachments to form
      if (filesArray.length > 0) {
        for (const file of filesArray) {
          form.append('files', file);
        }
      }
      // post comment form and check result was a success
      if (await postFileComments(sha256, form, setCommentError)) {
        setCommentError('Success');
        // fetch comment updates after successful posting
        fetchComments();
        // ensure we are on same page as new comment
        const newPageValue = Math.ceil((comments.length + 1) / PAGELIMIT) - 1;
        if (newPageValue != page && newPageValue != -1) {
          setPage(newPageValue);
        }
      }
    }
  };

  const AlertBanner = () => {
    return (
      <Fragment>
        {commentError == 'Success' && (
          <Alert className="attachment-card" variant="success">
            Comment has uploaded successfully!
          </Alert>
        )}
        {commentError != '' && commentError != 'Success' && (
          <Alert className="attachment-card" variant="danger">
            {commentError}
          </Alert>
        )}
      </Fragment>
    );
  };

  // CommentList component
  const CommentList = () => {
    return (
      <Fragment>
        {comments &&
          comments.slice(page * limit, page * limit + limit).map((singleCommentObj, i) => (
            <Card key={i} className="single-comment mb-2 panel">
              <Card.Header>
                {singleCommentObj.author} <i>{singleCommentObj.uploaded}</i>
              </Card.Header>
              <Card.Body>
                <Row key={i}>
                  <p key={i}>{singleCommentObj.comment}</p>
                </Row>
                {singleCommentObj &&
                  singleCommentObj.files &&
                  Object.keys(singleCommentObj.files).map((name, i) => (
                    <Col key={i}>
                      <a
                        href="#comments"
                        className="text"
                        onClick={() => getAttachment(singleCommentObj.id, name, singleCommentObj.files[name])}
                      >
                        {name}
                      </a>
                    </Col>
                  ))}
              </Card.Body>
            </Card>
          ))}
      </Fragment>
    );
  };

  return (
    <div id="comments-tab">
      <div className="comments">
        <CommentList />
      </div>
      {comments.length == 0 && (
        <Fragment>
          <Alert variant="" className="info">
            <Alert.Heading>
              <center>
                <h3>No Comments Available</h3>
              </center>
            </Alert.Heading>
            <center>
              <p>Be the first to leave a comment</p>
            </center>
          </Alert>
        </Fragment>
      )}
      <Row className="mt-4">
        <Col className="d-flex justify-content-center">
          {comments.length != 0 && (
            <Pagination>
              <Pagination.Prev onClick={() => setPage(page - 1)} disabled={page == 0} />
              <Pagination.Next onClick={() => setPage(page + 1)} disabled={page >= maxPage - 1} />
            </Pagination>
          )}
        </Col>
      </Row>
      <Row>
        <center>
          <Form.Control
            className="comment-entry"
            as="textarea"
            placeholder="Add Comment"
            onChange={(e) => setNewComment(e.target.value)}
            value={newComment}
          />
          <Row>
            <Col>
              <Card className="mt-2 panel attachment-card">
                <Card.Body className="d-flex justify-content-center">
                  <UploadDropzone zoneWidth={'100%'} setFiles={setFilesArray} />
                </Card.Body>
              </Card>
            </Col>
          </Row>
          <Row className="d-flex justify-content-center mt-2">
            <Col>
              <center>
                <AlertBanner />
              </center>
              <Button
                className="mt-3 primary-btn auto-width"
                onClick={() => handlePost(newComment, filesArray)}
                disabled={newComment ? false : true}
              >
                Post
              </Button>
            </Col>
          </Row>
        </center>
      </Row>
    </div>
  );
};
