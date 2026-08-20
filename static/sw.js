/* Service worker for terminal push notifications.
   Must be served from the SAME origin as the page, at the root path, or the
   browser will refuse to give it a scope covering the whole app. */

self.addEventListener("install", e => self.skipWaiting());
self.addEventListener("activate", e => e.waitUntil(self.clients.claim()));

self.addEventListener("push", event => {
  let data = {title: "Terminal alert", body: "", url: "/"};
  try{
    if(event.data) data = Object.assign(data, event.data.json());
  }catch{
    // A push with a non-JSON body should still surface rather than vanish.
    if(event.data) data.body = event.data.text();
  }
  event.waitUntil(
    self.registration.showNotification(data.title, {
      body: data.body,
      icon: "/icon-192.png",
      badge: "/icon-192.png",
      // Trade alerts are time-critical; do not let the OS silently collapse
      // them into a quiet group.
      requireInteraction: true,
      vibrate: [200, 100, 200],
      tag: data.tag || undefined,
      data: {url: data.url || "/"},
    })
  );
});

self.addEventListener("notificationclick", event => {
  event.notification.close();
  const target = (event.notification.data && event.notification.data.url) || "/";
  event.waitUntil(
    clients.matchAll({type: "window", includeUncontrolled: true}).then(list => {
      // Focus an existing tab if the terminal is already open, rather than
      // stacking up duplicates every time an alert is tapped.
      for(const c of list){
        if(c.url.includes(self.location.origin) && "focus" in c) return c.focus();
      }
      if(clients.openWindow) return clients.openWindow(target);
    })
  );
});
