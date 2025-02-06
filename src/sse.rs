use std::sync::Arc;
use std::{pin::Pin, task::Context, task::Poll, time::Duration};

use futures::Stream;
use ntex::web::{self, Error, HttpResponse};
use ntex::{time::interval, util::Bytes};
use parking_lot::Mutex;
use tokio::sync::mpsc::{channel, Receiver, Sender};

pub async fn new_client(broadcaster: web::types::State<Mutex<Broadcaster>>) -> HttpResponse {
    let rx = broadcaster.lock().new_client();

    HttpResponse::Ok()
        .header("content-type", "text/event-stream")
        .no_chunking()
        .streaming(rx)
}

pub struct Broadcaster {
    clients: Vec<Sender<Bytes>>,
}

impl Broadcaster {
    pub fn create() -> Arc<Mutex<Self>> {
        // Data ≃ Arc
        let me = Arc::new(Mutex::new(Broadcaster::new()));

        // ping clients every 10 seconds to see if they are alive
        Broadcaster::spawn_ping(me.clone());

        me
    }

    pub fn new() -> Self {
        Broadcaster {
            clients: Vec::new(),
        }
    }

    pub fn spawn_ping(me: Arc<Mutex<Self>>) {
        ntex::rt::spawn(async move {
            loop {
                let task = interval(Duration::from_secs(10));
                task.tick().await;
                me.lock().remove_stale_clients();
            }
        });
    }

    pub fn remove_stale_clients(&mut self) {
        let mut ok_clients = Vec::new();
        for client in self.clients.iter() {
            let result = client.clone().try_send(Bytes::from("data: ping\n\n"));

            if let Ok(()) = result {
                ok_clients.push(client.clone());
            }
        }
        self.clients = ok_clients;
    }

    pub fn new_client(&mut self) -> Client {
        let (tx, rx) = channel(100);

        tx.try_send(Bytes::from("data: connected\n\n")).unwrap();

        self.clients.push(tx);
        Client(rx)
    }

    pub fn send(&self, msg: &str) {
        let msg = Bytes::from(["data: ", msg, "\n\n"].concat());

        for client in self.clients.iter() {
            client.clone().try_send(msg.clone()).unwrap_or(());
        }
    }
}

// wrap Receiver in own type, with correct error type
pub struct Client(Receiver<Bytes>);

impl Stream for Client {
    type Item = Result<Bytes, Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.0).poll_recv(cx) {
            Poll::Ready(Some(v)) => Poll::Ready(Some(Ok(v))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}
