using Godot;
using System;
using System.Threading.Tasks;
using System.Text.Json;

public partial class OpçõesMeterologia : GridContainer
{
	// Called when the node enters the scene tree for the first time.
	public override void _Ready()
	{
		double latitude = -10.2128;
        double longitude = -48.3603;

        string url = $"https://open-meteo.com{latitude}&longitude={longitude}&current_weather=true";

        using HttpClient client = new HttpClient();
        
        HttpRequest httpRequest = GetNode<HttpRequest>("HTTPRequest");
        httpRequest.RequestCompleted += OnRequestCompleted;

        httpRequest.Request(url);
	}

	// Called every frame. 'delta' is the elapsed time since the previous frame.
	private void OnRequestCompleted(long result, long responseCode, string[] headers, byte[] body)
    {
        Godot.Collections.Dictionary json = Json.ParseString(Encoding.UTF8.GetString(body)).AsGodotDictionary();
        GD.Print(json["name"]);
    }
}
